"""订单流与筹码微观结构因子（Micro Structure）。

因子定义（W1~W8 均为初始化入参，便于超参数寻优）：
    OFSS = clip(Σ有效(w_i * x_i) / Σ有效w_i, -1, 1)
      cancel_score = clip((cancel_base - Cancel_Ratio) / cancel_base, -1, 1)
      OBI         最优买卖档队列失衡度（买一/卖一量差归一，[-1,1]）
      AR          攻击性买入比率（主动买盘占比，[-1,1]）
      Cancel_Ratio 撤单率（撤单量 / (撤单量+成交量)，[0,1]）
      BigFlow     大单驱动（大单净流 / 总成交额，[-1,1]）
    CPS = clip(Lock_Ratio*W5 + Accum_Delta*W6 + Panic_Ratio*W7 + Drift*W8, -1, 1)
      Lock_Ratio   主力锁仓比（近 N 日未换手沉淀筹码占比）
      Accum_Delta  筹码累积增量（近 N 日主动买净量 / 流通股本）
      Panic_Ratio  恐慌抛压比（低价区主动卖量 / 总主动卖量）
      Drift        筹码重心漂移（VWAP 的 N 日变化率）
    PSS = clip(W_body*实体分 + W_shadow*影线方向分
               + (1-W_body-W_shadow)*(2*窗口百分位-1), -1, 1)

防未来函数约定：
- OFSS / PSS 基于分钟级实时数据（当前 bar 已收盘），当日可用
- CPS 为日频分量（依赖当日完整成交），由 FeatureEngine 按 T-1 可用时点
  对齐后再 ffill 到分钟轴
- 筹码相关定义为主力行为的简化近似，可替换为精细筹码分布模型
"""

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.dataslice import SYMBOL, TRADE_DATE

logger = logging.getLogger("indicators.microstructure")

_EPS = 1e-12

def _drop_grouper_level(s):
    """groupby(...).rolling(...) 的结果索引为 (分组键, 原索引) MultiIndex，
    需剥离分组键层级后才能与原 DataFrame 对齐赋值。"""
    if isinstance(s.index, pd.MultiIndex):
        return s.reset_index(level=0, drop=True)
    return s


class MicroStructure:
    """订单流 / 筹码 / 价格结构因子。

    :param ofss_weights: OFSS 权重 (W1..W4)
    :param cps_weights:  CPS 权重 (W5..W8)
    :param cancel_base:  OFSS 中 (cancel_base - Cancel_Ratio) 基准，默认 0.2
    :param big_th:       BigFlow 的大单成交额阈值（元），默认 20 万
    :param pss_body_w:   PSS 中实体分的权重
    :param pss_shadow_w: PSS 中下影线减上影线的权重
    :param pss_window:   PSS 窗口百分位的窗口（分钟）
    :param chip_window:  筹码各分量的滚动窗口（交易日）
    :param panic_dev:    低价区阈值（价格 < vwap*(1-panic_dev) 视为恐慌抛售）
    """

    def __init__(
        self,
        ofss_weights: Sequence[float] = (0.3, 0.3, 0.2, 0.2),
        cps_weights: Sequence[float] = (0.3, 0.3, 0.2, 0.2),
        cancel_base: float = 0.2,
        big_th: float = 2e5,
        pss_body_w: float = 0.5,
        pss_shadow_w: float = 0.25,
        pss_window: int = 60,
        chip_window: int = 20,
        panic_dev: float = 0.02,
    ) -> None:
        """保存订单流与筹码因子的参数；权重采用统一配置入口，便于回测与优化保持相同口径。"""
        if len(ofss_weights) != 4 or len(cps_weights) != 4:
            raise ValueError("ofss_weights / cps_weights 必须为 4 个权重")
        if not (0.0 <= pss_body_w <= 1.0 and 0.0 <= pss_shadow_w
                and pss_body_w + pss_shadow_w <= 1.0):
            raise ValueError("PSS 实体与影线权重须非负且和不超过 1")
        self.w = {
            **dict(zip(("w1", "w2", "w3", "w4"), ofss_weights)),
            **dict(zip(("w5", "w6", "w7", "w8"), cps_weights)),
        }
        self.cancel_base = cancel_base
        self.big_th = big_th
        self.pss_body_w = pss_body_w
        self.pss_shadow_w = pss_shadow_w
        self.pss_window = pss_window
        self.chip_window = chip_window
        self.panic_dev = panic_dev

    # ------------------------------------------------------------------
    # OFSS：分钟级订单流成分
    # ------------------------------------------------------------------

    def ofss_components(self, ds) -> pd.DataFrame:
        """计算分钟级 OBI / AR / Cancel_Ratio / BigFlow。

        依赖 ds.l2_snapshot（快照）与 ds.tick_trades（逐笔）；
        OBI 使用分钟末买一/卖一队列量；任一档缺失或两档均无量时置 NaN。
        无撤单事件来源时 Cancel_Ratio 未知，不能当作已观测到零撤单。
        数据缺失时对应列置 NaN 并告警。
        :return: 长表 [symbol, obi, ar, cancel_ratio, big_flow]
        """
        parts: list = []

        if ds.l2_snapshot is not None and not ds.l2_snapshot.empty:
            snap = ds.l2_snapshot.copy()
            bid = pd.to_numeric(
                snap["bid1_v"] if "bid1_v" in snap else pd.Series(np.nan, index=snap.index),
                errors="coerce")
            ask = pd.to_numeric(
                snap["ask1_v"] if "ask1_v" in snap else pd.Series(np.nan, index=snap.index),
                errors="coerce")
            valid = (bid.notna() & ask.notna() & (bid >= 0) & (ask >= 0)
                     & ((bid + ask) > 0))
            snap["obi"] = ((bid - ask) / (bid + ask).where(valid)).clip(-1.0, 1.0)
            # 分钟末快照代表该分钟盘口状态
            obi = snap.groupby([snap.index, snap[SYMBOL]])["obi"].last().reset_index()
            obi = obi.rename(columns={"level_0": "ts"})
            parts.append(obi)
        else:
            logger.warning("缺少 l2_snapshot，OBI 置 NaN")

        if ds.tick_trades is None or ds.tick_trades.empty:
            logger.warning("缺少 tick_trades，AR / Cancel_Ratio / BigFlow 置 NaN")
        else:
            t = ds.tick_trades.copy()
            if "is_cancel" not in t.columns:
                t["is_cancel"] = False
            trade = t[~t["is_cancel"]]
            cancel = t[t["is_cancel"]]

            tr = trade.copy()
            tr["buy_vol"] = np.where(tr["side"] > 0, tr["volume"], 0.0)
            tr["sell_vol"] = np.where(tr["side"] < 0, tr["volume"], 0.0)
            agg = (
                tr.groupby([tr.index, tr[SYMBOL]])
                .agg(buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
                     turnover=("turnover", "sum"))
                .reset_index()
                .rename(columns={"level_0": "ts"})
            )
            agg["ar"] = (agg["buy_vol"] - agg["sell_vol"]) / (
                agg["buy_vol"] + agg["sell_vol"] + _EPS)
            if not cancel.empty:
                canc = (cancel.groupby([cancel.index, cancel[SYMBOL]])["volume"]
                        .sum().reset_index()
                        .rename(columns={"level_0": "ts", "volume": "cancel_vol"}))
                agg = agg.merge(canc, on=["ts", SYMBOL], how="left").fillna(0.0)
                agg["cancel_ratio"] = agg["cancel_vol"] / (
                    agg["cancel_vol"] + agg["buy_vol"] + agg["sell_vol"] + _EPS)
            else:
                logger.warning("逐笔数据无撤单事件，Cancel_Ratio 置 NaN")
                agg["cancel_ratio"] = np.nan
            parts.append(agg[["ts", SYMBOL, "ar", "cancel_ratio"]])

            bg = tr.copy()
            bg["big"] = np.where(bg["turnover"] >= self.big_th,
                                 bg["side"] * bg["turnover"], 0.0)
            g = (bg.groupby([bg.index, bg[SYMBOL]])
                 .agg(big_net=("big", "sum"), turnover=("turnover", "sum"))
                 .reset_index().rename(columns={"level_0": "ts"}))
            g["big_flow"] = g["big_net"] / (g["turnover"] + _EPS)
            parts.append(g[["ts", SYMBOL, "big_flow"]])

        if not parts:
            # l2_snapshot 与 tick_trades 均缺失：返回空成分表，
            # 下游左连接自然得到全 NaN，符合"缺失降级不中断"约定
            return pd.DataFrame(columns=["ts", SYMBOL, "obi", "ar",
                                         "cancel_ratio", "big_flow"])
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=["ts", SYMBOL], how="outer")
        return out.set_index("ts").sort_index()

    def ofss(self, comp) -> pd.Series:
        """由 OFSS 成分计算综合得分（缺项动态重归一化，量纲对齐到 [-1, 1]）。

        各分量均为 [-1, 1] 等比量纲：
            obi / ar / big_flow 已由 ofss_components 归一，
            cancel_score = clip((cancel_base - cancel_ratio)/cancel_base, -1, 1)
            （取消撤单率绝对差口径：cr=0 → +1，cr=2*cancel_base → -1）。
        权重动态取 self.w["w1".."w4"]，不做硬编码。
        分子 = Σ (有效特征 * 权重)，缺失项 fillna(0.0) 不贡献分子；
        分母 = Σ 有效权重；全缺失 → NaN；否则 clip(分子/分母, -1, 1)。
        """
        w1, w2, w3, w4 = (self.w["w1"], self.w["w2"],
                          self.w["w3"], self.w["w4"])
        cancel_score = np.clip(
            (self.cancel_base - comp["cancel_ratio"]) / self.cancel_base,
            -1.0, 1.0)
        factors = {
            "obi": comp["obi"],
            "ar": comp["ar"],
            "cancel_score": cancel_score,
            "big_flow": comp["big_flow"],
        }
        weights = {"obi": w1, "ar": w2, "cancel_score": w3, "big_flow": w4}

        numerator = 0.0
        denominator = pd.Series(0.0, index=comp.index, dtype=float)
        for name, factor in factors.items():
            valid = factor.notna()
            denominator = denominator + valid * weights[name]
            numerator = numerator + factor.fillna(0.0) * weights[name]
        denom_safe = denominator.replace(0.0, np.nan)  # 全缺失 → NaN
        return pd.Series(np.clip(numerator / denom_safe, -1.0, 1.0),
                         index=comp.index)

    # ------------------------------------------------------------------
    # CPS：日频筹码成分（T-1 可用时点对齐由调用方完成）
    # ------------------------------------------------------------------

    def chip_components(self, ds) -> pd.DataFrame:
        """计算日频筹码分量：Lock_Ratio / Accum_Delta / Panic_Ratio / Drift。

        依赖 ds.tick_trades 与 ds.kline（float_market_cap / vwap / volume）。
        近似筹码口径见类文档；返回 [trade_date, symbol, ...]。
        """
        if ds.tick_trades is None or ds.tick_trades.empty:
            logger.warning("缺少 tick_trades，筹码分量置 NaN")
            return pd.DataFrame(columns=[TRADE_DATE, SYMBOL, "lock_ratio",
                                         "accum_delta", "panic_ratio", "drift"])

        k = ds.kline.copy()
        k["day"] = k.index.normalize()
        # 日频 K 线：流通股本 = 流通市值 / 收盘价
        daily_k = k.groupby(["day", SYMBOL]).agg(
            close=("close", "last"), vwap=("vwap", "last"),
            volume=("volume", "sum"), mcap=("float_market_cap", "last")).reset_index()

        t = ds.tick_trades.copy()
        if "is_cancel" not in t.columns:
            t["is_cancel"] = False
        trade = t[~t["is_cancel"]]
        trade["day"] = trade.index.normalize()
        # 主动买/卖量、低价区主动卖量（< 当日 VWAP*阈值）
        trade = trade.merge(daily_k[["day", SYMBOL, "vwap"]], on=["day", SYMBOL],
                            how="left")
        trade["low_sell"] = np.where(
            (trade["side"] < 0) & (trade["price"] < trade["vwap"] * (1 - self.panic_dev)),
            trade["volume"], 0.0)
        trade["buy_vol"] = np.where(trade["side"] > 0, trade["volume"], 0.0)
        trade["sell_vol"] = np.where(trade["side"] < 0, trade["volume"], 0.0)
        day_t = trade.groupby(["day", SYMBOL]).agg(
            buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
            low_sell=("low_sell", "sum"),
        ).reset_index()

        df = daily_k.merge(day_t, on=["day", SYMBOL], how="left").fillna(0.0)
        df = df.sort_values([SYMBOL, "day"])
        df["shares"] = df["mcap"] / df["close"].replace(0, np.nan)
        df["turnover_r"] = df["volume"] / df["shares"].replace(0, np.nan)

        grp = df.groupby(SYMBOL, group_keys=False)
        lock_sum = _drop_grouper_level(
            grp["turnover_r"].rolling(self.chip_window, min_periods=1).sum())
        df["lock_ratio"] = (1 - lock_sum).clip(0, 1)
        accum = _drop_grouper_level(
            grp[["buy_vol", "sell_vol"]].rolling(self.chip_window, min_periods=1).sum())
        df["accum_delta"] = (accum.eval("(buy_vol - sell_vol)")
                             / df["shares"].replace(0, np.nan))
        sell_roll = _drop_grouper_level(
            grp["sell_vol"].rolling(self.chip_window, min_periods=1).sum())
        low_roll = _drop_grouper_level(
            grp["low_sell"].rolling(self.chip_window, min_periods=1).sum())
        df["panic_ratio"] = low_roll / (sell_roll + _EPS)
        df["drift"] = (df["vwap"] / grp["vwap"].shift(self.chip_window) - 1.0).clip(-1, 1)

        out = df.rename(columns={"day": TRADE_DATE})[
            [TRADE_DATE, SYMBOL, "lock_ratio", "accum_delta",
             "panic_ratio", "drift"]].copy()
        out[TRADE_DATE] = pd.to_datetime(out[TRADE_DATE])
        return out

    def cps(self, comp: pd.DataFrame) -> pd.Series:
        """筹码综合得分（量纲对齐到 [-1, 1] + 缺项动态重归一化）。

        四个分量统一 [-1, 1]：
            s_lock   = 2*lock_ratio - 1         锁仓 0%→+1，100%→-1
            s_accum  = clip(accum_delta / 0.05) 净吸筹，5% 经验基准撑满量纲
            s_panic  = clip(1 - 2*panic_ratio)  恐慌越高越空（修复旧公式越恐慌越加分）
            s_drift  = clip(drift / 0.15)        重心漂移，15% 经验基准
        权重动态取 self.w["w5".."w8"]，不硬编码。
        分子 = Σ 有效特征*权重；分母 = Σ 有效权重；全缺失(w_sum==0) → NaN；
        否则 cps = clip(分子/分母, -1, 1)。
        """
        w5, w6, w7, w8 = (self.w["w5"], self.w["w6"],
                          self.w["w7"], self.w["w8"])
        s_lock = 2.0 * comp["lock_ratio"] - 1.0
        s_accum = np.clip(comp["accum_delta"] / 0.05, -1.0, 1.0)
        s_panic = np.clip(1.0 - 2.0 * comp["panic_ratio"], -1.0, 1.0)
        s_drift = np.clip(comp["drift"] / 0.15, -1.0, 1.0)
        factors = [s_lock, s_accum, s_panic, s_drift]
        weights = [w5, w6, w7, w8]

        numerator = pd.Series(0.0, index=comp.index, dtype=float)
        denominator = pd.Series(0.0, index=comp.index, dtype=float)
        for factor, weight in zip(factors, weights):
            valid = factor.notna()
            denominator = denominator + valid * weight
            numerator = numerator + factor.fillna(0.0) * weight
        denom_safe = denominator.where(denominator > 0.0)  # w_sum==0 → NaN
        res = numerator / denom_safe
        return pd.Series(np.clip(res, -1.0, 1.0).values, index=comp.index)

    # ------------------------------------------------------------------
    # PSS：分钟级价格结构
    # ------------------------------------------------------------------

    def pss(self, kline: pd.DataFrame) -> pd.Series:
        """价格结构分：实体、影线方向与近 N 分钟窗口百分位。

        窗口统计包含当前已收盘的 bar（属历史），不引入未来信息。
        注意：kline 为含 symbol 列的长表（同一时间戳可有多行），索引非唯一，
        须用 groupby.transform（保持原行序）而非 Series 索引对齐运算。
        """
        k = kline.copy()
        span = k["high"] - k["low"]
        k["body"] = (k["close"] - k["open"]) / (span + _EPS)
        upper_shadow = (k["high"] - k[["open", "close"]].max(axis=1)) / (span + _EPS)
        lower_shadow = (k[["open", "close"]].min(axis=1) - k["low"]) / (span + _EPS)
        # 下影线较长表示低位承接；上影线较长表示高位抛压。
        wick_score = lower_shadow - upper_shadow
        grp = k.groupby(SYMBOL, group_keys=False)
        hi = grp["high"].transform(
            lambda s: s.rolling(self.pss_window, min_periods=2).max())
        lo = grp["low"].transform(
            lambda s: s.rolling(self.pss_window, min_periods=2).min())
        # 窗口无波动（一字/停牌）→ 位置分中性 0，杜绝因微小常数 eps 误判 -1.0
        is_flat = hi <= lo
        pct_score = np.where(is_flat, 0.0,
                             2.0 * (k["close"] - lo) / (hi - lo) - 1.0)
        pct_w = 1.0 - self.pss_body_w - self.pss_shadow_w
        score = pd.Series(0.0, index=k.index)
        if self.pss_body_w > 0.0:
            score = score + self.pss_body_w * k["body"]
        if self.pss_shadow_w > 0.0:
            score = score + self.pss_shadow_w * wick_score
        if pct_w > 0.0:
            score = score + pct_w * pct_score
        return score.clip(-1.0, 1.0)
