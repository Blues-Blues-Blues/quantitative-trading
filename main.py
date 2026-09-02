"""项目主入口：全链路冒烟测试（Smoke Test）。

用内置极简 Mock 数据（6 个交易日 × 2 只股票）跑通：
    数据构建 → 特征工程（Agent Profiling / Microstructure / Environment）
    → 多层闸门状态机（8 层开仓 / 6 层平仓）→ 回测撮合 → 绩效评估 → 可视化
并打印各环节耗时与状态检查信息。

剧情设计（保证 ≥1 笔完整 BUY→SELL 闭环）：
- D1~D4 强牛：每 Bar 超大单主动买 + 小单主动卖（Inst_Flow>0、CPS>0、OFSS>0 → S_push）
- D5~D6 转熊：每 Bar 超大单主动卖（Inst_Flow<0 → 状态机退出 S_push → 触发 SELL）
- 时序约束（T-1 对齐 + Next-Bar 执行，决定开/平仓时点）：
    * 日频因子（CPS/北向/RS/行业情绪）第 2 个交易日起才可用
    * GRS/Global_Mod 的滚动 z-score 需 20 根非 NaN 分钟 bar → D4 13:30 起有值
    * 故 BUY 落在 D4 14:00（Next-Bar 成交），SELL 落在 D5 10:00（T+1 解冻后，合规）

用法：
    python main.py
"""

import hashlib
import logging
import os
import pickle
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics.metrics import closed_trades, evaluate
from data.dataslice import DataSlice, SYMBOL, TRADE_DATE
from engine.backtest import BacktestEngine
from engine.execution import ExecutionCost
from engine.portfolio import Account
from engine.risk_control import PositionSizer
from indicators.basic import compute_all
from indicators.feature_engine import FEATURE_COLS, FeatureEngine
from indicators.microstructure import MicroStructure
from strategy.signals import (
    ACT_ADD,
    ACT_BUY,
    ACT_SELL,
    SignalSynthesizer,
    TradingStateMachine,
)

logger = logging.getLogger("main.smoke")

# ----------------------------------------------------------------------
# 冒烟 Mock 数据：6 个交易日，前 4 日强牛、后 2 日转熊
# ----------------------------------------------------------------------

SYMBOLS: Tuple[str, ...] = ("600000", "000001")
SYMBOL_TO_INDUSTRY: Dict[str, str] = {"600000": "银行", "000001": "银行"}

BULL_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
BEAR_DATES = ["2024-01-08", "2024-01-09"]
DATES = BULL_DATES + BEAR_DATES

# 冒烟显式参数（权重和为 1，全部落在 optimizer.SearchSpace 范围内；
# chip_window=1 保证第 2 个交易日起 CPS 有值；th_global_min=-0.8 匹配
# T-1 宏观对齐后 Global_Mod 到 D4 才有值的时序）
SMOKE_PARAMS: Dict[str, object] = {
    "weights": (0.35, 0.25, 0.25, 0.15),
    "th_ms_bull": 0.3, "th_ms_exit": -0.1, "th_lock": 0.4, "th_purity": 0.1,
    "th_global_min": -0.8, "th_adr_min": 0.3,
    "win_hold_max": 120, "inst_window": 1, "chip_window": 1,
}

INITIAL_CASH = 1e8

def _signals_cache_path(ds: DataSlice, params: Dict[str, object],
                        label: str) -> str:
    """信号落盘缓存路径：以 区间+标的+参数 为键（任一变化即失效）。"""
    meta = getattr(ds, "meta", {}) or {}
    key = hashlib.md5(
        f"{label}|{meta.get('start')}|{meta.get('end')}"
        f"|{','.join(sorted(meta.get('symbols', ['?'])))}"
        f"|{sorted((str(k), repr(v)) for k, v in params.items())}"
        .encode("utf-8")).hexdigest()[:16]
    return os.path.join("data", "feature_cache", f"signals_{key}.pkl")


def _window_end_str(params: Dict[str, object]) -> str:
    """reversal_window_end 透传：直接给定则原样；给定 reversal_window_span
    （09:30 起分钟数，寻优用）则换算为 HH:MM。"""
    if "reversal_window_end" in params:
        return str(params["reversal_window_end"])
    span = int(params.get("reversal_window_span", 30))
    total = 9 * 60 + 30 + span
    return f"{total // 60:02d}:{total % 60:02d}"

# 真实数据（data1+data2）回测区间与参数：
# （旧"放宽环境层闸门"（th_mrs_min/th_industry_min）已随二值化闸门废除而移除，
#   与寻优链路参数集合保持一致，保证寻优结果可在实盘复现。）
REAL_START = "2024-01-02"
REAL_END = "2024-12-31"
# 最优参数（当前 = 候选 #2 comparison）：seed=42 journal trial 11（验证段 Sharpe=+1.27 候选）
# 对照注记—生产默认 trial 14（训练段 top-1）：2024 单年 603019 Sharpe +1.03 / +295 万；
# 本组为 A/B 检验用，2024 单年 603019 结果见运行日志。窗口沿用寻优固定值 (1,1)。
REAL_PARAMS: Dict[str, object] = {
    # 组合权重（W_OFSS+W_CPS+W_INST+W_NORTH == 1，W_NORTH 归一化推导）
    "weights": (0.2069, 0.3547, 0.3981, 0.0403),
    # 窗口（寻优固定 (1,1)，与特征缓存签名一致）
    "inst_window": 1, "chip_window": 1,
    # ---- 连续评分：ES ----
    "w_es_ms": 0.2261, "w_es_purity": 0.3573, "w_es_mrs": 0.2324,
    "es_sigmoid_k": 3.2786, "th_es_entry": 0.2148,
    # ---- 连续评分：XS ----
    "th_xs_exit": -0.4482, "th_xs_reduce_high": 0.2893, "th_xs_crash": -0.6547,
    "w_xs_ms": 0.4213, "w_xs_purity": 0.2341,
    "w_xs_drawdown": 1.0 - 0.4213 - 0.2341,  # 归一化推导 = 0.3446
    # ---- 次日低开反包 ----
    "th_reversal_gap": -0.0472, "th_reversal_ofss": 0.1758,
    "reversal_add_mult": 0.7632, "reversal_window_span": 0,
    # ---- 连续评分：PS ----
    "base_decay_rate": 0.9097, "win_decay_grace": 31,
    "pnl_decay_profit_mult": 0.4506, "pnl_decay_loss_mult": 1.5633,
    "cancel_ratio_th": 0.3682, "fund_stability_penalty": 0.5913,
    # ---- 状态 / 一票否决 ----
    "th_retail_chase": 0.7961,
    # ---- 目标权重 Target_Weight（floor/cap 裁剪区间）----
    "base_weight": 0.1913, "reduce_step_ratio": 0.601,
    "tw_gmod_clip": (0.1947, 1.7618),
    "tw_cmod_clip": (0.6898, 1.2783),
    # ---- 撮合引擎（调仓死区）----
    "deadzone_th": 0.0831,
}
# 真实数据回测股票子集（空列表 = 全部 20 只；指定代码可大幅缩短运行时间）
# 单票检验：603019（计算机设备 / 中科曙光）
REAL_SYMBOLS: List[str] = ["603019"]


# ----------------------------------------------------------------------
# Mock 数据构造（仿 tests/test_optimizer.py，扩展为多标的）
# ----------------------------------------------------------------------

def cn_minutes(dates, freq: str = "30min") -> pd.DatetimeIndex:
    """A 股交易分钟轴（09:30~11:30 + 13:00~15:00）。"""
    parts = []
    for d in pd.to_datetime(dates):
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 09:30", f"{d:%Y-%m-%d} 11:30", freq=freq))
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 13:00", f"{d:%Y-%m-%d} 15:00", freq=freq))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])).sort_values()


def mk_kline(axis: pd.DatetimeIndex, bull_dates, symbols=SYMBOLS,
             bases=(10.0, 20.0), vol: float = 1e5) -> pd.DataFrame:
    """个股 K 线：牛段每 Bar 复利 +0.2%、熊段 -0.2%；放量 → amount > 1e7。"""
    bull = {pd.Timestamp(d).normalize() for d in bull_dates}
    per_sym = {}
    for sym, base in zip(symbols, bases):
        closes = [base]
        for t in axis[1:]:
            step = 0.002 if t.normalize() in bull else -0.002
            closes.append(closes[-1] * (1.0 + step))
        per_sym[sym] = closes
    rows = []
    for i, t in enumerate(axis):
        for sym in symbols:
            c = per_sym[sym][i]
            rows.append({
                "symbol": sym, "open": c * 0.995, "high": c * 1.01,
                "low": c * 0.99, "close": c, "volume": vol,
                "amount": c * vol * 100, "vwap": c, "float_market_cap": 1e9,
                "up_limit": c * 1.1, "down_limit": c * 0.9, "is_st": False,
            })
    df = pd.DataFrame(rows)
    df.index = axis.repeat(len(symbols))
    df.index.name = "ts"
    return df


def mk_snapshot(axis: pd.DatetimeIndex, symbols=SYMBOLS) -> pd.DataFrame:
    """五档 L2 快照：买盘量 > 卖盘量 → OBI > 0。"""
    rows = []
    for t in axis:
        for sym in symbols:
            row = {"symbol": sym}
            for i in range(1, 6):
                row[f"bid{i}_p"] = 10.0 - 0.01 * i
                row[f"bid{i}_v"] = 200.0 / i
                row[f"ask{i}_p"] = 10.0 + 0.01 * i
                row[f"ask{i}_v"] = 100.0 / i
            rows.append(row)
    df = pd.DataFrame(rows, index=axis.repeat(len(symbols)))
    df.index.name = "ts"
    return df


def mk_flow_ticks(axis: pd.DatetimeIndex, bull_dates, symbols=SYMBOLS) -> pd.DataFrame:
    """逐笔成交：牛段超大单买 2e6 + 小单卖 2e4（Inst>0、Retail<0）；
    熊段超大单卖 2e6（Inst<0 → 状态机退出 S_push → SELL）。"""
    bull = {pd.Timestamp(d).normalize() for d in bull_dates}
    rows = []
    for t in axis:
        for sym in symbols:
            if t.normalize() in bull:
                rows.append((t, sym, 10.0, 0, 2e6, 1, False))
                rows.append((t, sym, 10.0, 0, 2e4, -1, False))
            else:
                rows.append((t, sym, 10.0, 0, 2e6, -1, False))
    df = pd.DataFrame(rows, columns=["ts", "symbol", "price", "volume",
                                     "turnover", "side", "is_cancel"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")


def mk_index_min(axis: pd.DatetimeIndex) -> pd.DataFrame:
    """沪深300 指数：加速上涨（ma20 > ma60）→ MRS>0 且 个股跑赢 → RS>0。"""
    n = len(axis)
    closes = 3000.0 + 0.05 * np.arange(n) + 0.002 * np.arange(n) ** 2
    rows = []
    for i, t in enumerate(axis):
        c = closes[i]
        rows.append({"index_code": "000300.SH", "open": c, "high": c * 1.002,
                     "low": c * 0.998, "close": c,
                     "volume": 1e7 * (1.0 + 0.02 * i), "vwap": c,
                     "ma20": c, "ma60": c * 0.99})
    df = pd.DataFrame(rows, index=axis)
    df.index.name = "ts"
    return df


def mk_breadth(axis: pd.DatetimeIndex) -> pd.DataFrame:
    """全市场广度：ADR=2.0 达标（> 1.0），北向净流为正。"""
    df = pd.DataFrame(index=axis)
    df["advancers"] = 3000.0
    df["decliners"] = 1500.0
    df["adr"] = 2.0
    df["north_net"] = 5e7
    return df


def mk_industry(axis: pd.DatetimeIndex, name: str = "银行") -> pd.DataFrame:
    """行业资金流：单调递增 → IRS / Chain_Mod > 0。"""
    df = pd.DataFrame(index=axis)
    df["industry"] = name
    df["open"] = df["high"] = df["low"] = df["close"] = 1000.0
    df["volume"] = 1e6
    df["money_flow"] = np.linspace(0, 1, len(axis)) * 1e8
    return df


def mk_macro(dates, base: float = 100.0) -> pd.DataFrame:
    """全球宏观（日频，T-1 对齐）：美股/商品/美债均单调走强 → GRS>0。"""
    rows = []
    for i, d in enumerate(pd.to_datetime(dates)):
        rows.append({"trade_date": d, "us_spx": base + i, "us_ndx": base + i,
                     "us_dow": base + i, "brent": base + i, "gold": base + i,
                     "copper": base + i, "us10y": 3.0 + 0.01 * i, "dxy": 100 + i,
                     "hsi": base + i, "nky": base + i})
    return pd.DataFrame(rows)


def mk_north_margin(dates, symbols=SYMBOLS) -> pd.DataFrame:
    """北向 / 两融（日频，T+1 披露）：持仓递增 → North_Sync 偏多。"""
    idx = pd.to_datetime(dates)
    rows = []
    for i, d in enumerate(idx):
        for sym in symbols:
            rows.append({
                "symbol": sym, "trade_date": d,
                "north_holding": 1e7 + 2e5 * i,
                "north_buy_net": 1e6 if i % 3 == 0 else -5e5,
                "margin_fin_balance": 1e9 + 2e7 * i,
                "margin_sec_balance": 5e8 + 1e7 * i,
            })
    return pd.DataFrame(rows)


def mk_dragon_tiger(dates, symbols=SYMBOLS) -> pd.DataFrame:
    """龙虎榜（上榜日 trade_date，avail_date 由 aligner 标注为 T+1）。"""
    rows = []
    for d in pd.to_datetime(dates):
        for sym in symbols:
            rows.append({"symbol": sym, "trade_date": d,
                         "buy_amount": 1e8, "sell_amount": 5e7,
                         "net_amount": 5e7, "side": 1})
    return pd.DataFrame(rows)


def build_smoke_slice() -> DataSlice:
    """构造完整冒烟 DataSlice（龙虎榜经 TimeAligner 标注 T+1 可用日）。"""
    from data.aligner import TimeAligner
    axis = cn_minutes(DATES, freq="30min")
    ds = DataSlice(
        kline=mk_kline(axis, BULL_DATES),
        l2_snapshot=mk_snapshot(axis),
        tick_trades=mk_flow_ticks(axis, BULL_DATES),
        index_min=mk_index_min(axis),
        breadth=mk_breadth(axis),
        industry=mk_industry(axis),
        macro=mk_macro(DATES),
        north_margin=mk_north_margin(DATES),
        dragon_tiger=mk_dragon_tiger(DATES),
        meta={"symbols": list(SYMBOLS), "smoke": True},
    )
    # DataSlice.validate() 要求龙虎榜携带 avail_date，先按真实 T+1 规则标注
    ds.dragon_tiger = TimeAligner().align_dragon_tiger(ds.dragon_tiger, axis)
    return ds


# ----------------------------------------------------------------------
# 主流程：全链路冒烟测试
# ----------------------------------------------------------------------

def _tick(t0: float, label: str) -> float:
    dt = time.perf_counter() - t0
    logger.info("    └─ %s 耗时 %.3f s", label, dt)
    return time.perf_counter()


def run_pipeline(ds: DataSlice, params: Dict[str, object],
                 symbol_to_industry: Dict[str, str], label: str,
                 rank_gate=None) -> None:
    """全链路主流程（数据 → 特征 → 状态机 → 回测 → 绩效 → 绘图 → 检查）。"""
    logger.info("=" * 78)
    logger.info("%s全链路启动：%d 只股票（%s）", label, len(ds.symbols()),
                ds.meta.get("source", "?"))
    logger.info("=" * 78)
    start = time.perf_counter()
    t0 = start

    # ---- 环节 1：数据就绪 ----
    ds.validate()  # schema + 时间索引质量显式校验
    t0 = _tick(t0, f"数据就绪 + DataSlice.validate（{len(ds.kline)} 根 K 线）")

    # ---- 环节 2：特征工程（Agent Profiling / Microstructure / Environment）----
    fe = FeatureEngine(
        micro=MicroStructure(chip_window=int(params["chip_window"])),
        symbol_to_industry=symbol_to_industry,
    )
    features = fe.compute_cached(ds)
    feat_nan = int(features.isna().sum().sum())
    logger.info("特征工程输出：%d 行 × %d 列，NaN 元素 %d 个（T-1 对齐导致的预期缺失）",
                len(features), len(FEATURE_COLS), feat_nan)
    t0 = _tick(t0, "特征工程 FeatureEngine.compute_cached")

    # ---- 环节 3：信号合成 + 多层闸门状态机 ----
    syn = SignalSynthesizer(
        weights=tuple(params["weights"]),
        inst_window=int(params["inst_window"]),
        # 兼容参数（旧二值化闸门，决策链不再读取；SearchSpace 已停止采样，
        # 此处 .get 兜底仅保历史配置/外部脚本兼容）
        th_ms_bull=float(params.get("th_ms_bull", 0.0)),
        th_ms_exit=float(params.get("th_ms_exit", -0.1)),
        th_lock=float(params.get("th_lock", 0.5)),
        th_purity=float(params.get("th_purity", 0.0)),
        th_global_min=float(params.get("th_global_min", 0.0)),
        th_adr_min=float(params.get("th_adr_min", 1.0)),
        th_mrs_min=float(params.get("th_mrs_min", 0.0)),
        th_industry_min=float(params.get("th_industry_min", 0.0)),
        win_hold_max=int(params.get("win_hold_max", 240)),
        # ---- 连续评分 ES / XS / PS 与目标权重（与 StrategyOptimizer 口径一致，
        #      实盘参数含这些键时透传；SMOKE_PARAMS 缺省走默认值）----
        w_es_ms=float(params.get("w_es_ms", 0.4)),
        w_es_purity=float(params.get("w_es_purity", 0.3)),
        w_es_mrs=float(params.get("w_es_mrs", 0.3)),
        es_sigmoid_k=float(params.get("es_sigmoid_k", 3.0)),
        th_es_entry=float(params.get("th_es_entry", 0.4)),
        th_xs_exit=float(params.get("th_xs_exit", -0.3)),
        th_xs_reduce_high=float(params.get("th_xs_reduce_high", 0.2)),
        th_xs_crash=float(params.get("th_xs_crash", -0.6)),
        w_xs_ms=float(params.get("w_xs_ms", 0.5)),
        w_xs_purity=float(params.get("w_xs_purity", 0.3)),
        w_xs_drawdown=float(params.get("w_xs_drawdown", 0.2)),
        th_reversal_gap=float(params.get("th_reversal_gap", -0.015)),
        th_reversal_ofss=float(params.get("th_reversal_ofss", 0.2)),
        reversal_add_mult=float(params.get("reversal_add_mult", 0.5)),
        reversal_window_end=_window_end_str(params),
        base_decay_rate=float(params.get("base_decay_rate", 0.95)),
        win_decay_grace=int(params.get("win_decay_grace", 30)),
        pnl_decay_profit_mult=float(params.get("pnl_decay_profit_mult", 0.5)),
        pnl_decay_loss_mult=float(params.get("pnl_decay_loss_mult", 2.0)),
        cancel_ratio_th=float(params.get("cancel_ratio_th", 0.25)),
        fund_stability_penalty=float(params.get("fund_stability_penalty", 0.7)),
        th_retail_chase=float(params.get("th_retail_chase", 0.65)),
        base_weight=float(params.get("base_weight", 0.20)),
        reduce_step_ratio=float(params.get("reduce_step_ratio", 0.8)),
        tw_gmod_clip=tuple(
            float(x) for x in params.get("tw_gmod_clip", (0.2, 1.5))),
        tw_cmod_clip=tuple(
            float(x) for x in params.get("tw_cmod_clip", (0.5, 1.5))),
        symbol_to_industry=symbol_to_industry,
        rank_gate=rank_gate,
    )
    sm = TradingStateMachine(synthesizer=syn)
    cache_path = _signals_cache_path(ds, params, label)
    signals = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                signals = pickle.load(f)
            logger.info("信号缓存命中：%s（%d 个信号）", cache_path, len(signals))
        except Exception as exc:  # noqa: BLE001 缓存损坏则重算
            logger.warning("信号缓存读取失败，将重新计算：%s", exc)
            signals = None
    if signals is None:
        signals = sm.run(ds, features)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(signals, f)
            logger.info("信号缓存已写入：%s", cache_path)
        except Exception as exc:  # noqa: BLE001 缓存写失败不影响回测
            logger.warning("信号缓存写入失败（不影响回测）：%s", exc)
    act_cnt = Counter(s.action for s in signals)
    logger.info("状态机输出：%d 个信号 → %s", len(signals),
                {k: act_cnt[k] for k in (ACT_BUY, ACT_ADD, ACT_SELL, "HOLD")})
    t0 = _tick(t0, "信号合成 + 状态机（8/6 层闸门）")

    # ---- 环节 4：回测撮合引擎 ----
    engine = BacktestEngine(
        Account(initial_cash=INITIAL_CASH),
        ExecutionCost(),
        PositionSizer(),
        ds, signals,
        deadzone_th=float(params.get("deadzone_th", 0.05)),
    )
    trade_log, equity_curve = engine.run()
    filled = trade_log[trade_log["shares"] > 0]
    rejected = trade_log[trade_log["shares"] == 0]
    logger.info("回测撮合：%d 根 Bar 净值点；成交 %d 笔 / 拒绝 %d 笔；"
                "拒绝原因=%s", len(equity_curve), len(filled), len(rejected),
                dict(rejected["reason"].value_counts()) if len(rejected) else "无")
    t0 = _tick(t0, "回测撮合 BacktestEngine.run")

    # ---- 环节 5：绩效评估 ----
    metrics = evaluate(equity_curve, trade_log)
    trades = closed_trades(trade_log)
    logger.info("绩效评估：年化Sharpe=%.3f 最大回撤=%.2f%% 总盈亏=%+.0f 元 "
                "有效交易=%d 笔 胜率=%.0f%%",
                metrics["sharpe"], metrics["max_drawdown"] * 100,
                metrics["total_pnl"], metrics["n_trades"],
                metrics["win_rate"] * 100 if np.isfinite(metrics["win_rate"]) else float("nan"))
    for tr in trades:
        logger.info("    └─ %s 于 %s 买入 → %s 卖出，%d 股，盈亏 %+.0f 元",
                    tr["symbol"], tr["entry_ts"], tr["exit_ts"],
                    int(tr["shares"]), tr["pnl"])
    t0 = _tick(t0, "绩效评估 analytics.metrics.evaluate")

    # ---- 环节 6：可视化（analytics/plotter.py → analytics/pictures/）----
    chart_paths = _plot_smoke_charts(ds)
    t0 = _tick(t0, f"绘图 plot_trend（{len(chart_paths)} 张）")

    # ---- 状态检查 ----
    checks = _status_checks(ds, features, trade_log, equity_curve)
    logger.info("=" * 78)
    logger.info("状态检查汇总：")
    for ok, msg in checks:
        logger.info("    [%s] %s", "PASS" if ok else "FAIL", msg)
    if not all(ok for ok, _ in checks):
        raise SystemExit("冒烟测试未通过，请查看上方 FAIL 项")
    logger.info("=" * 78)
    logger.info("%s全链路全部通过 ✅（总耗时 %.3f s）", label,
                time.perf_counter() - start)
    logger.info("可视化输出：%s", ", ".join(chart_paths))


def run_smoke() -> None:
    """冒烟模式：内置 Mock 数据跑通全链路。"""
    run_pipeline(build_smoke_slice(), SMOKE_PARAMS,
                 SYMBOL_TO_INDUSTRY, "冒烟测试")


def run_real() -> None:
    """真实数据模式：data1（万得 L2）+ data2（日频 CSV）→ 全链路回测。

    回测标的由 REAL_SYMBOLS 指定（空列表 = 全部标的）。
    """
    from data.real_loader import RealDataLoader
    loader = RealDataLoader()
    symbols = loader.discover_symbols()
    if REAL_SYMBOLS:
        symbols = [s for s in symbols if s in REAL_SYMBOLS]
        logger.info("回测股票子集：%s（共 %d 只）", REAL_SYMBOLS, len(symbols))
    logger.info("真实数据发现 %d 只标的：%s", len(symbols),
                ", ".join(symbols[:6]) + ("..." if len(symbols) > 6 else ""))
    # 特征缓存预检（与 run_pipeline 中 FeatureEngine 构造参数一致）：
    # 命中 → 跳过逐笔成交/快照加载，特征表直接复用
    fe_probe = FeatureEngine(
        micro=MicroStructure(chip_window=int(REAL_PARAMS["chip_window"])),
        symbol_to_industry=loader.symbol_to_industry,
    )
    cache_hit = fe_probe.cache_exists(REAL_START, REAL_END, symbols)
    if cache_hit:
        logger.info("特征缓存预检命中：跳过 tick/l2_snapshot 加载（%s ~ %s）",
                    REAL_START, REAL_END)
    ds = loader.load_slice(symbols, REAL_START, REAL_END, skip_tick=cache_hit)
    run_pipeline(ds, REAL_PARAMS, loader.symbol_to_industry, "真实数据",
                 rank_gate=_build_rank_gate(REAL_PARAMS))


def _build_rank_gate(params: Dict[str, object]):
    """按 params 配置构造横截面排序闸门（未配置 → None，闸门关闭）。"""
    factor = params.get("rank_gate_factor")
    if not factor:
        return None
    from strategy.gates import CrossSectionalRankGate
    gate = CrossSectionalRankGate(
        factor=str(factor),
        top_quantile=float(params.get("rank_gate_top", 0.2)))
    logger.info("横截面排序闸门开启：%s 需处于昨日全池前 %.0f%%",
                gate.factor, gate.top_quantile * 100)
    return gate


def run_ic_flow(start: str, end: str,
                symbols: Optional[List[str]] = None) -> None:
    """真实数据横截面因子有效性分析（Rank IC / Normal IC / IC_IR / Q1~Q5 分层）。

    需要全股票池（横截面）与逐笔数据（合成因子依赖 tick 因子），
    未命中特征缓存时计算并落盘，后续重复分析直接复用。
    """
    from analytics.ic_analyzer import run_ic_analysis
    from data.real_loader import RealDataLoader
    from pathlib import Path

    loader = RealDataLoader()
    pool = symbols or loader.discover_symbols()
    logger.info("=" * 78)
    logger.info("横截面因子有效性分析：%d 只股票（%s ~ %s）", len(pool), start, end)
    logger.info("=" * 78)

    fe = FeatureEngine(
        micro=MicroStructure(chip_window=int(REAL_PARAMS["chip_window"])),
        symbol_to_industry=loader.symbol_to_industry,
    )
    if not fe.cache_exists(start, end, pool):
        logger.info("特征缓存未命中，需加载逐笔数据计算（%s ~ %s）", start, end)
    ds = loader.load_slice(pool, start, end,
                           skip_tick=fe.cache_exists(start, end, pool))
    features = fe.compute_cached(ds)

    res = run_ic_analysis(
        ds, features, loader.symbol_to_industry, REAL_PARAMS,
        stock_basic=loader.macro.load_stock_basic_full())

    ic = res["ic_summary"]
    logger.info("— IC 汇总（Rank IC / IC_IR / t 值 / 胜率）：")
    for _, r in ic.iterrows():
        logger.info("  %-13s %-7s IC=%+.4f IC_IR=%+.3f t=%+5.2f 胜率=%3.0f%% 期数=%d",
                    r["factor"], r["horizon"], r["rank_ic"], r["ic_ir"],
                    r["t_stat"], r["win_rate"] * 100, r["n_days"])

    q = res["quantile_summary"]
    logger.info("— Q1~Q5 分层（日频等权，fwd_1d 与 fwd_30m）：")
    for _, r in q.iterrows():
        logger.info("  %-13s %-7s Q1=%+.4f Q2=%+.4f Q3=%+.4f Q4=%+.4f Q5=%+.4f "
                    "spread=%+.4f 单调=%+.2f",
                    r["factor"], r["horizon"], r["Q1"], r["Q2"], r["Q3"],
                    r["Q4"], r["Q5"], r["spread"], r["monotonic"])

    out_dir = Path("analytics") / "pictures"
    out_dir.mkdir(parents=True, exist_ok=True)
    ic.to_csv(out_dir / "ic_summary.csv", index=False)
    q.to_csv(out_dir / "quantile_summary.csv", index=False)
    logger.info("IC 结果已导出：%s, %s",
                out_dir / "ic_summary.csv", out_dir / "quantile_summary.csv")


def _plot_smoke_charts(ds: DataSlice) -> List[str]:
    """把分钟 K 线按日聚合后跑 compute_all，再调用 plot_trend 出图。"""
    from analytics.plotter import plot_trend
    paths = []
    for sym in ds.symbols():
        k = ds.kline[ds.kline[SYMBOL] == sym].copy()
        daily = k.resample("D").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"),
            amount=("amount", "sum"),
        )
        daily = compute_all(daily)  # ma5/ma20/is_trend 等绘图必需列
        daily["code"] = sym
        fname = os.path.join("analytics", "pictures", f"smoke_{sym}.png")
        path = plot_trend(daily, fname=fname,
                          title=f"{sym} 冒烟回测段（黄色区域=震荡区间）")
        paths.append(path)
    return paths


def _status_checks(ds: DataSlice, features: pd.DataFrame,
                   trade_log: pd.DataFrame, equity_curve: pd.DataFrame) -> List[Tuple[bool, str]]:
    """冒烟状态检查：闭环完整性 / 未来函数 / T+1 / 空引用 / 资金安全。"""
    checks: List[Tuple[bool, str]] = []

    # 1) 必须有 ≥1 笔完整 BUY→SELL 闭环
    filled = trade_log[trade_log["shares"] > 0]
    n_closed = len(closed_trades(trade_log))
    checks.append((n_closed >= 1,
                   f"存在 ≥1 笔完整平仓交易（实际 {n_closed} 笔，成交 {len(filled)} 笔）"))

    # 2) 无未来函数：特征表时间轴不早于日频因子可用日。
    #    FeatureEngine 已在龙虎榜路径显式调用 TimeAligner.verify_no_lookahead，
    #    此处对全部日频因子做二次回归校验（特征行不得早于 macro/north T-1 可用日）。
    checks.append((
        _verify_no_lookahead(ds, features),
        "无未来函数（龙虎榜 T+1 断言 + 日频因子 T-1 对齐二次校验）"))

    # 3) 无 T+1 违规：当日卖出总量不得超过当日开盘可卖持仓
    #   （当日买入份额必须冻结；允许"当日卖旧仓 + 当日买新仓"——
    #    旧口径按「当日有买入又有卖出」粗判会误报）。
    sells = filled[filled["side"] == "SELL"]
    t1_ok = True
    t1_detail: Optional[str] = None
    t_trace: Dict[str, List[str]] = {}

    def _trace(sym: str, s: str) -> None:
        lst = t_trace.setdefault(sym, [])
        lst.append(s)
        del lst[:-20]

    hold: Dict[str, int] = {}          # 累计持仓（按成交重放）
    open_hold: Dict[str, int] = {}     # 当日开盘可卖 = 前一日收盘持仓
    sold_today: Dict[str, int] = {}    # 当日已卖出
    cur_day: Optional[pd.Timestamp] = None
    for _, tr in filled.sort_values("ts").iterrows():
        sym = str(tr["symbol"])
        day = pd.Timestamp(tr["ts"]).normalize()
        if day != cur_day:
            cur_day, open_hold, sold_today = day, dict(hold), {}
        side = str(tr["side"])
        shares = int(tr["shares"])
        if side in ("BUY", "ADD"):
            hold[sym] = hold.get(sym, 0) + shares
            _trace(sym, f"{tr['ts']} {side}+{shares}→hold{hold[sym]}")
        elif side == "SELL":
            sold_today[sym] = sold_today.get(sym, 0) + shares
            if sold_today[sym] > open_hold.get(sym, 0):
                t1_ok = False
                rows = filled[(filled["symbol"] == sym)]
                rows = rows[rows.index <= tr.name].sort_values("ts")
                rows_text = "; ".join(
                    f"{r['ts']} {r['side']} {int(r['shares'])}"
                    for _, r in rows.tail(40).iterrows())
                t1_detail = (f"{tr['ts']} {sym} SELL {shares} 股，当日已卖 "
                             f"{sold_today[sym]} > 开盘可卖 "
                             f"{int(open_hold.get(sym, 0))}（{sym} 成交轨迹: {rows_text}）")
                break
            hold[sym] = hold.get(sym, 0) - shares
            _trace(sym, f"{tr['ts']} {side}-{shares}→hold{hold[sym]}"
                        f"/日卖{sold_today[sym]}")
    checks.append((t1_ok, "无 T+1 违规（当日卖出 ≤ 当日开盘可卖持仓）"
                   + ("" if t1_ok else f"；首违：{t1_detail}")))

    # 4) 无空指针/空引用：关键产物非空且列完整
    checks.append((len(equity_curve) > 0 and {"cash", "total_equity"} <=
                   set(equity_curve.columns), "净值曲线非空且列完整"))

    # 5) 资金安全：现金非负、净值非负
    cash_min = float(equity_curve["cash"].min())
    equity_min = float(equity_curve["total_equity"].min())
    checks.append((cash_min >= -1e-6 and equity_min > 0,
                   f"资金安全（现金最小值 {cash_min:,.0f}，净值最小值 {equity_min:,.0f}）"))
    return checks


def _verify_no_lookahead(ds: DataSlice, features: pd.DataFrame) -> bool:
    """日频因子 T-1 可见性二次校验。

    口径：T-1 对齐（allow_exact_matches=False）下，A 股时点 T 只能使用
    外部记录日期严格早于 T 的数据。若数据表含早于窗口起始的历史记录
    （真实数据通常如此），首日出现日频值是合法的 T-1 值；只有当天
    无法追溯到更早记录时出现值，才是未来函数。
    """
    first_day = pd.Timestamp(features.index.normalize()[0])
    first_mask = features.index.normalize() == first_day
    daily_cols = ["north_sync", "margin_pressure", "cps", "dt_net"]
    for col in daily_cols:
        if col not in features.columns:
            continue
        v = features.loc[first_mask, col]
        if not pd.notna(v).any():
            continue
        if not _has_prior_daily_record(ds, col, first_day):
            return False
    return True


def _has_prior_daily_record(ds: DataSlice, col: str, first_day: pd.Timestamp) -> bool:
    """日频因子 col 在 first_day 出现值，是否有严格早于 first_day 的可用记录。"""
    if col in ("north_sync", "margin_pressure"):
        table = ds.north_margin
    elif col == "dt_net":
        table = ds.dragon_tiger
    else:  # cps：由窗口内 tick/kline 计算并经 T-1 对齐，窗口首日不可能有更早数据
        return False
    if table is None or table.empty or TRADE_DATE not in table.columns:
        return False
    return bool((pd.to_datetime(table[TRADE_DATE]) < first_day).any())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="量化交易全链路入口")
    ap.add_argument("--data", choices=("smoke", "real"), default="smoke",
                    help="数据来源：smoke=内置 Mock；real=data1+data2 真实数据")
    ap.add_argument("--analyze-ic", action="store_true",
                    help="真实数据横截面因子有效性分析（IC/IC_IR/Q1~Q5 分层）")
    ap.add_argument("--ic-start", default=REAL_START, help="IC 分析起始日")
    ap.add_argument("--ic-end", default=REAL_END, help="IC 分析结束日")
    args = ap.parse_args()
    # 统一日志目录（data/logs 已在 .gitignore 排除）：控制台 + 追加写入 run.log，
    # 目录不存在自动创建；日志格式与内容与原来一致。
    _log_dir = os.path.join("data", "logs")
    os.makedirs(_log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(_log_dir, "run.log"),
                                      encoding="utf-8")],
    )
    if args.data == "real":
        if args.analyze_ic:
            run_ic_flow(args.ic_start, args.ic_end)
        else:
            run_real()
    else:
        run_smoke()
