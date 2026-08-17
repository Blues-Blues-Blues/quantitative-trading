"""信号合成公式与多层闸门交易状态机（SignalSynthesizer / TradingStateMachine）。

核心公式：
    Agent_MS       = W_OFSS*OFSS + W_CPS*CPS + W_INST*sign(Inst_Flow)
                     + W_NORTH*North_Sync          （W 之和 == 1.0，强制断言）
    Final_MS       = (Agent_MS + Chain_Mod) * (1.0 + Global_Mod)
    Capital_Purity = 0.35*sign(Inst_Flow) + 0.25*North_Sync
                     - 0.2*sign(Retail_Flow) - 0.2*sign(Youzi_Flow)

第二步因子集中不存在的变量，在本模块内按现有因子近似（窗口/权重全部参数化）：
    Retail_Chase  近 N 分钟零售累计净流 / 全体资金流绝对值累计，clip [0,1]
    RS            个股近 N 交易日累计收益 - 沪深300 近 N 交易日累计收益（日频）
    Industry_MS   行业近 N 交易日资金流累计变化率（日频）

状态机（逐 Bar / 逐标的）：
    S_push        多证据进攻态：Final_MS>0 且 OFSS>0 且 Inst_Flow>0 且 CPS>0
    S_youzi_only  游资主导 + 散户盲从：Youzi_Flow>0 且 Inst_Flow<=0
                  且 Retail_Chase > youzi_chase_th（默认 0.6），禁止开仓
    S_noise       其余（默认态），禁止开仓

开仓 8 层闸门（全部为真）与平仓 6 层闸门（任一为真）见 SignalSynthesizer。

防未来函数约定：
- 分钟级合成（Agent/Final/Purity/Chase）只用当前及历史 Bar
- RS / Industry_MS 为日频因子，经 T-1 asof 对齐后才进入分钟轴（当日不可见）
- 状态机按时间升序逐 Bar 推进；持仓状态（入场 VWAP / 持仓分钟数）
  在 Bar 内部递增，绝不使用未来 Bar
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.aligner import TimeAligner
from data.dataslice import SYMBOL, TRADE_DATE, DataSlice

logger = logging.getLogger("strategy.signals")

_EPS = 1e-12

# ---- 状态常量 ----
S_PUSH = "S_push"
S_YOUZI_ONLY = "S_youzi_only"
S_NOISE = "S_noise"
_STATES = (S_PUSH, S_YOUZI_ONLY, S_NOISE)

# ---- 动作常量 ----
ACT_BUY = "BUY"
ACT_ADD = "ADD"
ACT_SELL = "SELL"
ACT_HOLD = "HOLD"

# ---- 合成所需的最小特征列 ----
REQUIRED_FEATURES: List[str] = [
    "ofss", "cps", "inst_flow", "north_sync", "retail_flow", "youzi_flow",
    "chain_mod", "global_mod", "grs", "mrs", "irs", "lock_ratio",
]


def _drop_grouper_level(s):
    """剥离 groupby(...).rolling(...) 结果中的 (分组键, 原索引) MultiIndex 层级。"""
    if isinstance(s.index, pd.MultiIndex):
        return s.reset_index(level=0, drop=True)
    return s


@dataclass
class Signal:
    """单根 Bar / Tick 的交易信号对象。

    :param symbol:    标的代码
    :param timestamp: 决策时点（Bar 时间戳）
    :param action:    BUY / ADD / SELL / HOLD
    :param state:     S_push / S_youzi_only / S_noise
    :param metrics:   决策指标快照（合成分 + 闸门明细），用于复盘
    """

    symbol: str
    timestamp: pd.Timestamp
    action: str
    state: str
    metrics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol, "timestamp": self.timestamp,
            "action": self.action, "state": self.state, "metrics": self.metrics,
        }


@dataclass
class Position:
    """状态机内部模拟持仓（paper position）。"""

    symbol: str
    entry_time: pd.Timestamp
    entry_vwap: float
    last_price: float = 0.0
    bars_held: int = 0          # 已持仓的分钟数（当前 Bar 计入）
    last_add_bar: int = 0       # 最近一次加仓时的 bars_held


class SignalSynthesizer:
    """信号合成与闸门评估（无状态纯函数集）。

    所有权重与阈值均为初始化入参，便于 Optuna 超参数寻优。
    权重约束：W_OFSS + W_CPS + W_INST + W_NORTH == 1.0（构造时断言）。
    """

    def __init__(
        self,
        weights: Sequence[float] = (0.35, 0.25, 0.25, 0.15),  # (W_OFSS, W_CPS, W_INST, W_NORTH)
        chase_window: int = 30,        # Retail_Chase 滚动窗口（分钟）
        rs_window: int = 20,           # RS 相对强度窗口（交易日）
        industry_window: int = 20,     # Industry_MS 窗口（交易日）
        inst_window: int = 1,          # Inst_Flow 平滑窗口（分钟，1 = 不平滑）
        youzi_chase_th: float = 0.6,   # S_youzi_only 的 Retail_Chase 阈值（规则硬约束）
        # ---- 开仓闸门阈值 ----
        th_global_min: float = 0.0,    # ① 全球层
        th_adr_min: float = 1.0,       # ② 系统层 ADR
        th_mrs_min: float = 0.0,       # ③ 系统层 MRS
        th_industry_min: float = 0.0,  # ④ 产业层
        th_ms_bull: float = 0.0,       # ⑥ Final_MS 多头阈值
        th_lock: float = 0.5,          # ⑥ Main_Lock_Ratio
        th_chase: float = 0.7,         # ⑥ Retail_Chase
        th_purity: float = 0.0,        # ⑥ Capital_Purity
        th_amount: float = 1e7,        # ⑦ 分钟成交额（元）
        start_time: str = "09:45",     # ⑧ 开仓时间窗
        end_time: str = "14:50",
        # ---- 平仓闸门阈值 ----
        th_ms_exit: float = -0.1,      # ② Final_MS 退出阈值
        th_slippage: float = 0.03,     # ④ 入场 VWAP 移动止损（3%）
        win_hold_max: int = 240,       # ⑤ 最大持仓分钟数
        circuit_index_drop: float = 0.015,  # ⑥ 指数盘中跌破 VWAP 比例（1.5%）
        th_grs_circuit: float = -1.5,  # ⑥ 全球隔夜熔断阈值
        symbol_to_industry: Optional[Dict[str, str]] = None,
    ) -> None:
        if len(weights) != 4:
            raise ValueError(f"weights 必须为 4 个权重 (W_OFSS, W_CPS, W_INST, W_NORTH)，当前: {weights}")
        total = float(np.sum(weights))
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"权重之和必须等于 1.0，当前和: {total}")
        if win_hold_max <= 0:
            raise ValueError(f"win_hold_max 必须为正整数，当前: {win_hold_max}")
        if inst_window < 1:
            raise ValueError(f"inst_window 必须 >= 1，当前: {inst_window}")

        self.w_ofss, self.w_cps, self.w_inst, self.w_north = map(float, weights)
        self.chase_window = chase_window
        self.rs_window = rs_window
        self.industry_window = industry_window
        self.inst_window = inst_window
        self.youzi_chase_th = youzi_chase_th

        self.th_global_min = th_global_min
        self.th_adr_min = th_adr_min
        self.th_mrs_min = th_mrs_min
        self.th_industry_min = th_industry_min
        self.th_ms_bull = th_ms_bull
        self.th_lock = th_lock
        self.th_chase = th_chase
        self.th_purity = th_purity
        self.th_amount = th_amount
        self.start_time = pd.Timestamp(start_time).time()
        self.end_time = pd.Timestamp(end_time).time()

        self.th_ms_exit = th_ms_exit
        self.th_slippage = th_slippage
        self.win_hold_max = win_hold_max
        self.circuit_index_drop = circuit_index_drop
        self.th_grs_circuit = th_grs_circuit
        self.symbol_to_industry = dict(symbol_to_industry or {})

    # ------------------------------------------------------------------
    # 信号合成（列级）
    # ------------------------------------------------------------------

    def _agent_ms(self, features: pd.DataFrame) -> pd.Series:
        """主体情绪分：加权合成，缺失分量按剩余权重重归一化，全缺失 → NaN。"""
        parts = pd.DataFrame({
            "ofss": features["ofss"].astype(float),
            "cps": features["cps"].astype(float),
            "inst": np.sign(features["inst_flow"]),
            "north": features["north_sync"].astype(float),
        })
        w = np.array([self.w_ofss, self.w_cps, self.w_inst, self.w_north])
        mask = parts.notna()
        wsum = (mask * w).sum(axis=1)
        score = (parts.fillna(0.0).to_numpy() * w).sum(axis=1)
        # 全 NaN 时 wsum==0 → 保持 NaN
        return pd.Series(np.where(wsum > 0, score / np.where(wsum > 0, wsum, np.nan),
                                  np.nan), index=features.index)

    def _retail_chase(self, features: pd.DataFrame) -> pd.Series:
        """散户追涨度：近 N 分钟零售累计净流 / 全体资金流绝对值累计，clip [0,1]。"""
        f = features.sort_index()
        # DataFrameGroupBy 不支持 abs()，先在分组前计算绝对流
        f = f.assign(_ret_abs=f["retail_flow"].abs(),
                     _inst_abs=f["inst_flow"].abs(),
                     _youzi_abs=f["youzi_flow"].abs())
        g = f.groupby(SYMBOL, group_keys=False)
        num = g["retail_flow"].rolling(self.chase_window, min_periods=1).sum()
        den = (g[["_ret_abs", "_inst_abs", "_youzi_abs"]]
               .rolling(self.chase_window, min_periods=1).sum().sum(axis=1))
        s = (num / (den + _EPS)).clip(0.0, 1.0)
        # groupby 结果为 (symbol, ts) 顺序；多标的时须重排为 features 的行顺序，
        # 否则与 DatetimeIndex（含重复 ts）无法逐行对齐
        if isinstance(s.index, pd.MultiIndex):
            key = pd.MultiIndex.from_arrays(
                [features[SYMBOL].to_numpy(), features.index.to_numpy()])
            s = s.reindex(key)
            # 恢复 features 的 DatetimeIndex（含重复 ts）以便逐行赋值
            s = pd.Series(s.to_numpy(), index=features.index)
        return s

    def _relative_strength(self, ds: DataSlice, axis: pd.DatetimeIndex) -> pd.DataFrame:
        """个股相对强度 RS（日频，T-1 对齐）：个股近 N 日收益 - 指数近 N 日收益。

        返回长表 [ts, symbol, rs]；缺数据 → rs 全 NaN（闸门保守关闭）。
        """
        if ds.kline is None or ds.kline.empty or ds.index_min is None or ds.index_min.empty:
            return pd.DataFrame(columns=["ts", SYMBOL, "rs"])

        k = ds.kline.copy()
        k["day"] = k.index.normalize()
        dk = k.groupby(["day", SYMBOL])["close"].last().reset_index()
        dk = dk.rename(columns={"day": TRADE_DATE})
        # 首日无前收盘 → 中性 0（不因缺乏历史而错误阻断/放行）
        dk["ret_s"] = dk.groupby(SYMBOL)["close"].pct_change().fillna(0.0)

        idx = ds.index_min.copy()
        idx["day"] = idx.index.normalize()
        di = idx.groupby(["day", "index_code"])["close"].last().reset_index()
        # 全市场以首个指数代码代表
        first_code = idx["index_code"].iloc[0]
        di = di[di["index_code"] == first_code].rename(
            columns={"day": TRADE_DATE, "close": "idx_close"})
        di["ret_i"] = di["idx_close"].pct_change().fillna(0.0)

        dk = dk.merge(di[[TRADE_DATE, "ret_i"]], on=TRADE_DATE, how="left")
        dk["ret_i"] = dk["ret_i"].fillna(0.0)
        dk["drs"] = dk["ret_s"] - dk["ret_i"]
        dk["rs"] = _drop_grouper_level(
            dk.groupby(SYMBOL, group_keys=False)["drs"]
            .rolling(self.rs_window, min_periods=1).sum())
        return self._align_daily_to_minute(dk[[TRADE_DATE, SYMBOL, "rs"]], axis, ["rs"])

    def _industry_ms(self, ds: DataSlice, axis: pd.DatetimeIndex) -> pd.DataFrame:
        """行业层情绪 Industry_MS（日频，T-1 对齐）：行业资金流近 N 日累计变化率。

        按行业计算后映射到个股（symbol_to_industry）；无映射 → NaN。
        返回长表 [ts, symbol, industry_ms]。
        """
        if ds.industry is None or ds.industry.empty or not self.symbol_to_industry:
            return pd.DataFrame(columns=["ts", SYMBOL, "industry_ms"])

        ind = ds.industry.copy()
        ind["day"] = ind.index.normalize()
        di = ind.groupby(["day", "industry"])["money_flow"].sum().reset_index()
        di = di.rename(columns={"day": TRADE_DATE})
        di["flow_ret"] = di.groupby("industry")["money_flow"].pct_change().fillna(0.0)
        di["ims"] = _drop_grouper_level(
            di.groupby("industry", group_keys=False)["flow_ret"]
            .rolling(self.industry_window, min_periods=1).sum())

        # 行业级对齐（无 symbol 列 → 按 industry 分组）
        rows = []
        aligner = TimeAligner()
        for ind_name, g in di.groupby("industry"):
            a = aligner.align_external(g, axis, ["ims"], date_col=TRADE_DATE)
            a["industry"] = ind_name
            rows.append(a)
        if not rows:
            return pd.DataFrame(columns=["ts", SYMBOL, "industry_ms"])
        ind_long = pd.concat(rows).reset_index().rename(columns={"index": "ts"})

        # 映射到个股
        map_df = pd.DataFrame({
            SYMBOL: list(self.symbol_to_industry.keys()),
            "industry": list(self.symbol_to_industry.values()),
        })
        out = ind_long.merge(map_df, on="industry", how="inner")
        return out[["ts", SYMBOL, "ims"]].rename(columns={"ims": "industry_ms"})

    @staticmethod
    def _align_daily_to_minute(daily: pd.DataFrame, axis: pd.DatetimeIndex,
                               value_cols: List[str]) -> pd.DataFrame:
        """日频长表按 symbol 分组做 T-1 asof 对齐，返回分钟长表 [ts, symbol, *value_cols]。"""
        rows = []
        aligner = TimeAligner()
        for sym, g in daily.groupby(SYMBOL):
            a = aligner.align_external(g, axis, value_cols, date_col=TRADE_DATE)
            a[SYMBOL] = sym
            rows.append(a)
        if not rows:
            return pd.DataFrame(columns=["ts", SYMBOL] + value_cols)
        out = pd.concat(rows)
        return out.reset_index().rename(columns={"index": "ts"})

    # ------------------------------------------------------------------
    # 综合入口
    # ------------------------------------------------------------------

    def synthesize(self, ds: DataSlice, features: pd.DataFrame) -> pd.DataFrame:
        """在特征表上追加全部合成列与近似变量列。

        返回 DataFrame 的 index 与 features 一致（DatetimeIndex），
        新增列：agent_ms / final_ms / capital_purity / retail_chase / rs / industry_ms。
        """
        missing = [c for c in REQUIRED_FEATURES if c not in features.columns]
        if missing:
            raise ValueError(f"features 缺少合成必需列: {missing}")
        if not features.index.is_monotonic_increasing:
            raise ValueError("features 索引未按时间升序，请先对齐排序")

        out = features.copy()
        axis = features.index

        # WIN_INST：Inst_Flow 平滑窗口（按标的滚动均值，transform 保持原行序）
        if self.inst_window > 1:
            out["inst_flow"] = out.groupby(SYMBOL)["inst_flow"].transform(
                lambda s: s.rolling(self.inst_window, min_periods=1).mean())

        agent = self._agent_ms(out)
        out["agent_ms"] = agent
        out["final_ms"] = self.final_ms(
            agent, out["chain_mod"], out["global_mod"])
        out["capital_purity"] = self.capital_purity(out)
        out["retail_chase"] = self._retail_chase(out)

        # 日频近似变量：T-1 对齐到分钟轴后左连接（无值时保持 NaN）
        rs = self._relative_strength(ds, axis)
        ims = self._industry_ms(ds, axis)
        out = out.reset_index().rename(columns={"index": "ts"})
        if not rs.empty:
            out = out.merge(rs, on=["ts", SYMBOL], how="left")
        else:
            out["rs"] = np.nan
        if not ims.empty:
            out = out.merge(ims, on=["ts", SYMBOL], how="left")
        else:
            out["industry_ms"] = np.nan
        return out.set_index("ts")

    def final_ms(self, agent_ms: pd.Series, chain_mod: pd.Series,
                 global_mod: pd.Series) -> pd.Series:
        """最终市场情绪分 = (Agent_MS + Chain_Mod) * (1 + Global_Mod)。

        Chain_Mod / Global_Mod 缺失视为中性 0（不惩罚也不奖励）。
        """
        chain = chain_mod.fillna(0.0).astype(float)
        gmod = global_mod.fillna(0.0).astype(float)
        return (agent_ms.astype(float) + chain) * (1.0 + gmod)

    def capital_purity(self, features: pd.DataFrame) -> pd.Series:
        """资金纯净度：机构/北向流入加分，散户/游资流入减分。"""
        return (0.35 * np.sign(features["inst_flow"])
                + 0.25 * features["north_sync"].fillna(0.0)
                - 0.2 * np.sign(features["retail_flow"])
                - 0.2 * np.sign(features["youzi_flow"]))

    # ------------------------------------------------------------------
    # 状态判定（无状态纯函数）
    # ------------------------------------------------------------------

    def state_of(self, row: pd.Series) -> str:
        """逐行判定状态机状态（S_youzi_only 优先级最高，其次 S_push，默认 S_noise）。"""
        youzi_only = (row["youzi_flow"] > 0) and (row["inst_flow"] <= 0) \
            and (row["retail_chase"] > self.youzi_chase_th)
        if youzi_only:
            return S_YOUZI_ONLY
        push = (row["final_ms"] > 0) and (row["ofss"] > 0) \
            and (row["inst_flow"] > 0) and (row["cps"] > 0)
        if push:
            return S_PUSH
        return S_NOISE

    # ------------------------------------------------------------------
    # 开仓闸门（8 层，全部为真）
    # ------------------------------------------------------------------

    def entry_gates(self, row: pd.Series) -> Dict[str, bool]:
        """8 层开仓闸门明细。NaN 比较恒为 False（保守关闭闸门）。"""
        return {
            # ① 全球层
            "global": self._gt(row, "global_mod", self.th_global_min),
            # ② 系统层：沪深300 MA20 > MA60 且 ADR 达标
            "system": bool(self._gt(row, "ma20", row.get("ma60", np.nan))
                           and self._gt(row, "adr", self.th_adr_min)),
            # ③ 系统层：MRS
            "beta": self._gt(row, "mrs", self.th_mrs_min),
            # ④ 产业层
            "industry": self._gt(row, "irs", self.th_industry_min),
            # ⑤ Alpha 层：个股相对强度与行业情绪
            "alpha": bool(self._gt(row, "rs", 0.0)
                          and self._gt(row, "industry_ms", 0.0)),
            # ⑥ 个股层
            "stock": bool(
                row.get("state", S_NOISE) == S_PUSH
                and self._gt(row, "final_ms", self.th_ms_bull)
                and self._gt(row, "lock_ratio", self.th_lock)
                and row.get("retail_chase", 1.0) < self.th_chase
                and self._gt(row, "capital_purity", self.th_purity)),
            # ⑦ 流动性与合规
            "liquidity": bool(
                self._gt(row, "amount", self.th_amount)
                and not bool(row.get("is_st", True))
                and self._gt(row, "up_limit", row.get("close", np.nan))
                and self._gt(row, "close", row.get("down_limit", np.nan))),
            # ⑧ 时间窗口
            "time": self._in_time_window(row),
        }

    def entry_all(self, row: pd.Series) -> bool:
        """开仓闸门：全部为真才允许开仓/加仓。"""
        return all(self.entry_gates(row).values())

    # ------------------------------------------------------------------
    # 平仓闸门（6 层，任一为真）
    # ------------------------------------------------------------------

    def exit_triggers(self, row: pd.Series, pos: Position) -> Dict[str, bool]:
        """6 层平仓闸门明细（依赖持仓上下文：入场 VWAP / 持仓分钟数）。"""
        # ⑥ 熔断：指数盘中跌破 VWAP 或全球隔夜暴跌
        idx_circuit = bool(
            self._lt(row, "index_close", row.get("index_vwap", np.nan)
                     * (1.0 - self.circuit_index_drop)))
        grs_circuit = bool(self._lt(row, "grs", self.th_grs_circuit))
        return {
            "state": row.get("state", S_NOISE) in (S_NOISE, S_YOUZI_ONLY),
            "ms": bool(self._lt(row, "final_ms", self.th_ms_exit)),
            "purity": bool(self._lt(row, "capital_purity", 0.0)),
            "stop": bool(self._lt(row, "close",
                                   pos.entry_vwap * (1.0 - self.th_slippage))),
            "hold_time": pos.bars_held > self.win_hold_max,
            "circuit": idx_circuit or grs_circuit,
        }

    def exit_any(self, row: pd.Series, pos: Position) -> bool:
        """平仓闸门：任一为真即触发卖出。"""
        return any(self.exit_triggers(row, pos).values())

    # ------------------------------------------------------------------
    # 内部小工具
    # ------------------------------------------------------------------

    @staticmethod
    def _gt(row: pd.Series, left: str, right: float) -> bool:
        """严格大于，NaN 安全（NaN > x / x > NaN 均为 False）。"""
        return bool(pd.notna(row.get(left)) and pd.notna(right)
                    and row[left] > right)

    @staticmethod
    def _lt(row: pd.Series, left: str, right: float) -> bool:
        """严格小于，NaN 安全。"""
        return bool(pd.notna(row.get(left)) and pd.notna(right)
                    and row[left] < right)

    def _in_time_window(self, row: pd.Series) -> bool:
        """⑧ 当前时间在 [start_time, end_time] 内。"""
        t = row.get("ts", row.name)
        try:
            cur = pd.Timestamp(t).time()
        except Exception:
            return False
        return self.start_time <= cur <= self.end_time


class TradingStateMachine:
    """有限状态机 + 多层闸门信号决策管道（有状态）。

    逐 Bar / 逐标的推进：
    - 状态判定（S_push / S_youzi_only / S_noise）由 SignalSynthesizer.state_of 完成
    - 开仓：无持仓且 S_push 且 8 层闸门全过 → BUY
    - 加仓：已持仓且 S_push 且 8 层闸门全过 且 距上次加仓 >= min_add_interval → ADD
    - 平仓：已持仓且 6 层闸门任一为真 → SELL
    - 其余 → HOLD（每根 Bar / 每标的均输出一个 Signal）

    内部维护 paper positions（入场时间 / 入场 VWAP / 持仓分钟数），
    供平仓闸门 ④⑤ 使用；连续回测可跨 run() 调用保留，也可 reset()。
    """

    def __init__(
        self,
        synthesizer: Optional[SignalSynthesizer] = None,
        min_add_interval: int = 5,   # 两次加仓之间的最小 Bar 数
        add_requires_entry_gates: bool = True,  # 加仓是否复用全部开仓闸门
    ) -> None:
        self.syn = synthesizer or SignalSynthesizer()
        if min_add_interval < 0:
            raise ValueError(f"min_add_interval 不能为负，当前: {min_add_interval}")
        self.min_add_interval = min_add_interval
        self.add_requires_entry_gates = add_requires_entry_gates
        self.positions: Dict[str, Position] = {}
        self.reset()

    def reset(self) -> None:
        """清空内部持仓（新回测/新交易日前调用）。"""
        self.positions = {}
        self._step = 0  # 全局 Bar 计数（仅用于排序断言）

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, ds: DataSlice, features: pd.DataFrame) -> List[Signal]:
        """逐 Bar 生成信号。

        :param ds:       对齐后的 DataSlice（提供 kline/index_min/breadth 供合规与市场闸门）
        :param features: FeatureEngine 输出的特征长表（DatetimeIndex + symbol + 特征列）
        :return: 按 (timestamp, symbol) 升序的 Signal 列表
        """
        ds.validate()
        ev = self._build_eval_table(ds, features)

        signals: List[Signal] = []
        prev_ts: Optional[pd.Timestamp] = None
        for ts, grp in ev.groupby("ts"):
            cur = pd.Timestamp(ts)
            # 状态机严格按时间升序推进，绝不回退
            if prev_ts is not None:
                assert cur >= prev_ts, f"状态机收到乱序时间戳: {cur} < {prev_ts}"
            prev_ts = cur

            for row in grp.itertuples(index=False, name=None):
                row = pd.Series(dict(zip(ev.columns, row)))
                sym = row[SYMBOL]
                pos = self.positions.get(sym)
                if pos is not None:
                    signals.append(self._on_holding(row, pos))
                else:
                    signals.append(self._on_flat(row))
        return signals

    # ------------------------------------------------------------------
    # 内部：评估表构建
    # ------------------------------------------------------------------

    def _build_eval_table(self, ds: DataSlice, features: pd.DataFrame) -> pd.DataFrame:
        """特征表 + 合成列 + kline/index/breadth 市场列 → 行级评估表。"""
        ev = self.syn.synthesize(ds, features)
        ev = ev.reset_index().rename(columns={"index": "ts"})

        # 个股合规 / 价格列（⑦ 与止损）
        k = ds.kline.copy().reset_index().rename(columns={"index": "ts"})
        k_cols = ["ts", SYMBOL, "close", "vwap", "amount",
                  "up_limit", "down_limit", "is_st"]
        k = k[[c for c in k_cols if c in k.columns]]
        ev = ev.merge(k, on=["ts", SYMBOL], how="left")

        # 市场列（②⑥）：取首个指数代码
        if ds.index_min is not None and not ds.index_min.empty:
            idx = ds.index_min.copy()
            code = idx["index_code"].iloc[0]
            idx = idx[idx["index_code"] == code].reset_index().rename(
                columns={"index": "ts", "close": "index_close",
                         "vwap": "index_vwap"})
            ev = ev.merge(idx[["ts", "index_close", "index_vwap", "ma20", "ma60"]],
                          on="ts", how="left")
        # ADR（②）
        if ds.breadth is not None and not ds.breadth.empty and "adr" in ds.breadth.columns:
            br = ds.breadth[["adr"]].reset_index().rename(columns={"index": "ts"})
            ev = ev.merge(br, on="ts", how="left")

        # 状态列（逐行无状态判定）
        ev["state"] = ev.apply(self.syn.state_of, axis=1)
        assert ev["ts"].is_monotonic_increasing, "评估表必须按时间升序"
        return ev

    # ------------------------------------------------------------------
    # 内部：逐行决策
    # ------------------------------------------------------------------

    def _on_flat(self, row: pd.Series) -> Signal:
        """无持仓：S_push 且全部开仓闸门通过 → BUY，否则 HOLD。"""
        gates = self.syn.entry_gates(row)
        if self.syn.entry_all(row):
            assert row["state"] == S_PUSH
            self.positions[row[SYMBOL]] = Position(
                symbol=row[SYMBOL], entry_time=pd.Timestamp(row["ts"]),
                entry_vwap=float(row["vwap"]), last_price=float(row["close"]),
                bars_held=0, last_add_bar=0)
            return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_BUY,
                          row["state"], self._metrics(row, entry_gates=gates))
        return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_HOLD,
                      row["state"], self._metrics(row, entry_gates=gates))

    def _on_holding(self, row: pd.Series, pos: Position) -> Signal:
        """已持仓：先递增持仓分钟数，再评估平仓 / 加仓。"""
        pos.bars_held += 1
        pos.last_price = float(row["close"])

        triggers = self.syn.exit_triggers(row, pos)
        if self.syn.exit_any(row, pos):
            del self.positions[row[SYMBOL]]
            return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_SELL,
                          row["state"], self._metrics(row, exit_triggers=triggers))

        # 加仓：S_push 且（可选）全部开仓闸门通过 且 满足最小加仓间隔
        add_ok = (row["state"] == S_PUSH)
        if self.add_requires_entry_gates:
            add_ok = add_ok and self.syn.entry_all(row)
        if add_ok and (pos.bars_held - pos.last_add_bar >= self.min_add_interval):
            pos.last_add_bar = pos.bars_held
            return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_ADD,
                          row["state"], self._metrics(row))
        return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_HOLD,
                      row["state"], self._metrics(row))

    @staticmethod
    def _metrics(row: pd.Series,
                 entry_gates: Optional[Dict[str, bool]] = None,
                 exit_triggers: Optional[Dict[str, bool]] = None) -> Dict[str, object]:
        """信号附带的指标快照（合成分 + 触发的闸门明细）。"""
        m: Dict[str, object] = {}
        for col in ("agent_ms", "final_ms", "capital_purity", "retail_chase",
                    "rs", "industry_ms", "ofss", "cps", "inst_flow",
                    "north_sync", "lock_ratio", "mrs", "irs", "grs",
                    "global_mod", "chain_mod"):
            v = row.get(col, np.nan)
            m[col] = float(v) if pd.notna(v) else None
        if entry_gates is not None:
            m["entry_gates"] = entry_gates
        if exit_triggers is not None:
            m["exit_triggers"] = exit_triggers
        return m

    # ------------------------------------------------------------------
    # 输出工具
    # ------------------------------------------------------------------

    @staticmethod
    def to_frame(signals: Sequence[Signal]) -> pd.DataFrame:
        """Signal 列表 → DataFrame（timestamp, symbol, action, state 列 + 展开指标）。"""
        rows = [s.to_dict() for s in signals]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "symbol", "action", "state"])
        out = pd.DataFrame(rows)
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        # 固定前 4 列：timestamp, symbol, action, state，其余（metrics 等）紧随其后
        base_cols = ["timestamp", "symbol", "action", "state"]
        rest = [c for c in out.columns if c not in base_cols]
        return out[base_cols + rest].sort_values(
            ["timestamp", "symbol"]).reset_index(drop=True)