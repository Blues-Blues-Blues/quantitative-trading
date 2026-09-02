"""信号合成公式与连续评分交易状态机（SignalSynthesizer / TradingStateMachine）。

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

连续评分（废弃二值化开平仓闸门的信号判定，全部系数可寻优）：
    入场分 ES = sigmoid(w_es_ms*Final_MS + w_es_purity*Capital_Purity
                        + w_es_mrs*MRS_c) ∈ [0, 1]
        Final_MS / Capital_Purity 已严格在 [-1, 1]，直接使用，缺失 → 0
        MRS_c      = clip(MRS, ±mrs_clip) / mrs_clip                 （有界化）
        S_youzi_only 硬掩码否决：ES = 0.0（游资独舞直接清零）
    持仓分 PS = ES * Time_Decay * Fund_Stability ∈ [0, 1]
        Time_Decay：前 win_decay_grace 分钟为保护期 = 1.0（防刚入场被误判减仓）；
            此后按浮盈亏非对称衰减（effective_bars = bars_held - win_decay_grace）：
            pnl_ratio = (close - avg_cost) / avg_cost
            浮盈(pnl_ratio>0)：factor = 1-(1-base_decay_rate)*pnl_decay_profit_mult
            浮亏(pnl_ratio<=0)：factor = 1-(1-base_decay_rate)*pnl_decay_loss_mult
            Time_Decay = clip(factor ** (effective_bars / time_decay_interval),
                              0.1, 1.0)
        Fund_Stability = 撤单率 > cancel_ratio_th 或盘口变薄 → 惩罚系数，否则 1.0
    出局分 XS = w_xs_ms*Final_MS + w_xs_purity*Capital_Purity
                - w_xs_drawdown*Drawdown_From_High ∈ [-1, 1]
        一票否决（强制 XS = -1.0）：big_flow<0 且 Retail_Chase>th_retail_chase
        （游资溃逃）；或沪深300 日内跌破 VWAP*(1-circuit_index_drop)（大盘跳水）
    目标权重 Target_Weight ∈ [0, max_single_position]（驱动撮合引擎差额调仓，
        见 engine.backtest 的动作→目标权重映射）：
        未持仓且硬过滤全过且 ES>=th_es_entry
            = base_weight * ES * clip(1+Global_Mod, tw_gmod_clip)
                                * clip(1+Chain_Mod, tw_cmod_clip)，否则 0
        持仓（XS 四分判定链）：
            XS >= th_xs_reduce_high      → base_weight * PS * 乘子（正常持仓调仓）
            th_xs_exit < XS < th_xs_reduce_high
                                         → simulated_weight * reduce_step_ratio
                                            （容错阶梯减仓，不清仓）
            th_xs_crash < XS <= th_xs_exit → 0（常规清仓）
            XS <= th_xs_crash 或一票否决   → 0（极速清仓 Crash / Panic Exit）
        simulated_weight 由状态机动作维护（BUY 建仓 / DECAY_REDUCE 触发 ×0.8 /
        ADD 重算），与引擎真实成交仓位相互独立。

决策链路（TradingStateMachine 逐 Bar）：
    前置：A 股硬过滤层（ST 禁买 / 涨跌停禁买卖 / 交易时间窗 / 成交额门槛），
          一票否决条件，全部与连续评分无关，独立生效。
    未持仓：硬过滤全过 且 ES >= th_es_entry → BUY
    持仓（XS 四分判定链）：
        XS >= th_xs_reduce_high          → ADD（S_push 且满足间隔）/ HOLD
        th_xs_exit < XS < th_xs_reduce_high
                                         → DECAY_REDUCE（容错阶梯减仓，
                                            ×reduce_step_ratio，受
                                            min_reduce_interval 节奏约束）
        th_xs_crash < XS <= th_xs_exit   → SELL（常规清仓）
        XS <= th_xs_crash 或一票否决      → SELL（极速清仓 Crash / Panic Exit）

历史接口兼容（不删除，避免破坏 attribution / 旧测试语义）：
    entry_gates / entry_all / exit_triggers / exit_any 保留，内部映射为
    新评分 + 硬过滤层的包装（键不变，语义见各方法 docstring）。

防未来函数约定：
- 分钟级合成（Agent/Final/Purity/Chase）只用当前及历史 Bar
- RS / Industry_MS 为日频因子，经 T-1 asof 对齐后才进入分钟轴（当日不可见）
- 状态机按时间升序逐 Bar 推进；持仓状态（入场 VWAP / 加权成本 / 持仓最高价 /
  持仓分钟数）在 Bar 内部递增，绝不使用未来 Bar
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.aligner import TimeAligner
from data.dataslice import SYMBOL, TRADE_DATE, DataSlice
from strategy.gates import CrossSectionalRankGate, add_previous_rank_columns

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
ACT_DECAY_REDUCE = "DECAY_REDUCE"   # 阶梯减仓（XS 回落区间）

# 资金流无状态平滑基准（tanh 归一化，无滚动窗口 / 时序累积依赖）
_DEFAULT_FLOW_SCALE = 1e7  # 默认 1000 万基准量

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


def _norm_flow(flow_series: pd.Series,
               amount_series: Optional[pd.Series] = None) -> np.ndarray:
    """无状态资金流平滑：norm_flow = tanh(flow / scale)，量纲有界到 (-1, 1)。

    彻底移除 rolling_std 标准化（不再有常量大流→0 误判、开盘前 N 根由
    min_periods 导致的钝化盲区、突发反转时 std 剧增引发的信号滞后——
    全部是时序累积 / 滚动窗口状态不良特性，此处全部无状态化）。

    scale 基准：
      - 提供 amount_series：scale = clip(amount * 0.05, 1e6, 1e8)，逐 bar 按
        成交额自适应规模（amount 缺失的 bar 回退到默认 1e7）；
      - 未提供 amount_series：scale = 1e7（默认 1000 万基准量）。

    flow 缺失 → 输出 NaN（交由上层缺失权重重归一化处理，不强行置 0）。
    无任何滚动窗口、无时序累积依赖、无开盘前 N 根盲区。
    """
    flow = flow_series.astype(float)
    if amount_series is None:
        scale = np.full(len(flow), _DEFAULT_FLOW_SCALE, dtype=float)
    else:
        amt = amount_series.astype(float).to_numpy()
        computed = np.clip(amt * 0.05, 1e6, 1e8)
        scale = np.where(np.isnan(computed), _DEFAULT_FLOW_SCALE, computed)
    return np.tanh(flow.to_numpy() / scale)


@dataclass
class Signal:
    """单根 Bar / Tick 的交易信号对象。

    :param symbol:    标的代码
    :param timestamp: 决策时点（Bar 时间戳）
    :param action:    BUY / ADD / SELL / DECAY_REDUCE / HOLD
    :param state:     S_push / S_youzi_only / S_noise
    :param metrics:   决策指标快照（合成分 + 评分 + 闸门明细），用于复盘
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
    """状态机内部模拟持仓（paper position）。

    high_price_watermark : 持仓期最高价（逐 Bar 更新），用于回撤与一票否决
    avg_cost             : 加权成本价（入场 = 首 Bar vwap；加仓时按市价重算），
                           用于动量豁免安全垫与回撤基准
    simulated_weight     : 信号层模拟目标权重（非真实成交）。BUY 建仓时 =
                           开仓目标权重；DECAY_REDUCE 触发时 ×reduce_step_ratio；
                           ADD 触发时按持仓分重算；SELL 时随持仓删除置 0。
                           供 generate_target_weights 的「阶梯减仓目标」取当前值。
    """

    symbol: str
    entry_time: pd.Timestamp
    entry_vwap: float
    last_price: float = 0.0
    bars_held: int = 0          # 已持仓的分钟数（当前 Bar 计入）
    last_add_bar: int = 0       # 最近一次加仓时的 bars_held
    last_reduce_bar: int = 0    # 最近一次阶梯减仓时的 bars_held
    high_price_watermark: float = 0.0
    avg_cost: float = 0.0
    simulated_weight: float = 0.0


class SignalSynthesizer:
    """信号合成与连续评分（无状态纯函数集）。

    所有权重与阈值均为初始化入参，便于 Optuna 超参数寻优。
    权重约束：W_OFSS + W_CPS + W_INST + W_NORTH == 1.0（构造时断言）。
    评分参数：w_es_* / w_xs_* / time_decay_* / th_es_entry / th_xs_* 等，
    默认值见构造签名，全部可被 SearchSpace 采样。
    """

    def __init__(
        self,
        weights: Sequence[float] = (0.35, 0.25, 0.25, 0.15),  # (W_OFSS, W_CPS, W_INST, W_NORTH)
        chase_window: int = 30,        # Retail_Chase 滚动窗口（分钟）
        rs_window: int = 20,           # RS 相对强度窗口（交易日）
        industry_window: int = 20,     # Industry_MS 窗口（交易日）
        inst_window: int = 1,          # Inst_Flow 平滑窗口（分钟，1 = 不平滑）
        th_retail_chase: float = 0.65, # S_youzi_only 与 XS 一票否决共用的追涨阈值
        youzi_chase_th: Optional[float] = None,  # 旧参数名兼容（映射到 th_retail_chase）
        # ---- A 股硬过滤层（一票否决前置条件，独立于评分）----
        th_amount: float = 1e7,        # ⑦ 分钟成交额（元）
        start_time: str = "10:00",     # ⑧ 开仓时间窗（10:00 整之前禁买，SELL/减仓不受限）
        end_time: str = "14:50",
        # ---- 入场分 ES ----
        w_es_ms: float = 0.4,          # Final_MS 权重
        w_es_purity: float = 0.3,      # Capital_Purity 权重
        w_es_mrs: float = 0.3,         # MRS 权重
        mrs_clip: float = 3.0,         # MRS 有界化半宽（clip ±3 后 /3）
        es_sigmoid_k: float = 3.0,     # sigmoid 陡度
        th_es_entry: float = 0.4,      # 开仓 ES 门槛
        # ---- 持仓分 PS ----
        base_decay_rate: float = 0.95,     # 基准衰减率（每 time_decay_interval 分钟）
        time_decay_interval: float = 10.0, # 衰减计算周期（分钟）
        win_decay_grace: int = 30,         # 衰减保护期（前 N 分钟 Time_Decay=1.0）
        pnl_decay_profit_mult: float = 0.5,  # 浮盈态衰减倍率（默认减半 → 0.975）
        pnl_decay_loss_mult: float = 2.0,    # 浮亏态衰减倍率（默认加倍 → 0.90）
        cancel_ratio_th: float = 0.25,    # 撤单率阈值（超过 → 资金不稳定）
        fund_stability_penalty: float = 0.7,  # 资金不稳定惩罚系数
        obi_thin_th: float = 0.05,      # 盘口变薄：|OBI| 阈值（近似）
        big_thin_th: float = 0.05,      # 盘口变薄：|big_flow| 阈值（近似）
        # ---- 出局分 XS ----
        w_xs_ms: float = 0.5,          # Final_MS 权重
        w_xs_purity: float = 0.3,      # Capital_Purity 权重
        w_xs_drawdown: float = 0.2,    # 回撤权重
        th_xs_exit: float = -0.3,          # XS <= 此值 → 常规清仓
        th_xs_reduce_high: float = 0.2,    # XS >= 此值 → 正常持仓按 PS 调仓
        th_xs_crash: float = -0.6,         # XS <= 此值（或一票否决）→ 极速清仓
        # ---- 次日低开反包（Reversal / Counter-Attack，已持仓承接加仓）----
        th_reversal_gap: float = -0.015,  # 深度低开/下探跌幅门槛（<= 此值触发）
        th_reversal_ofss: float = 0.2,    # 盘口承接分 OFSS 门槛（委买聚集+主动买盘）
        reversal_add_mult: float = 0.5,   # 反包加仓系数（增量 = base*ES*mult）
        reversal_window_end: str = "10:00",  # 反包判定截止时间（受开仓 time 闸门共同约束）
        reduce_fraction: float = 0.5,  # 旧减仓比例参数（保留构造兼容；新决策统一用 reduce_step_ratio）
        circuit_index_drop: float = 0.015,  # 指数盘中跌破 VWAP 比例（1.5%）
        # ---- 目标权重 Target_Weight（驱动撮合引擎差额调仓）----
        base_weight: float = 0.20,   # 开仓/调仓基准权重（与 PositionSizer.base_position 同源）
        max_single_position: float = 0.30,  # 目标权重单股上限（与 Account/PositionSizer 同值）
        reduce_step_ratio: float = 0.8,     # 阶梯减仓目标比例（Target = simulated × 0.8）
        tw_gmod_clip: Tuple[float, float] = (0.2, 1.5),  # (1+Global_Mod) 乘子裁剪区间
        tw_cmod_clip: Tuple[float, float] = (0.5, 1.5),  # (1+Chain_Mod) 乘子裁剪区间
        # ---- 保留兼容参数（旧二值化闸门阈值，新决策不再使用）----
        th_global_min: float = 0.0,
        th_adr_min: float = 1.0,
        th_mrs_min: float = 0.0,
        th_industry_min: float = 0.0,
        th_ms_bull: float = 0.0,
        th_lock: float = 0.5,
        th_chase: float = 0.7,
        th_purity: float = 0.0,
        th_ms_exit: float = -0.1,
        th_slippage: float = 0.03,
        win_hold_max: int = 240,
        th_grs_circuit: float = -1.5,
        symbol_to_industry: Optional[Dict[str, str]] = None,
        # ---- 横截面排序闸门（可选，默认关闭）----
        rank_gate: Optional["CrossSectionalRankGate"] = None,
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
        for name, val in (("w_es_ms", w_es_ms), ("w_es_purity", w_es_purity),
                          ("w_es_mrs", w_es_mrs)):
            if val < 0:
                raise ValueError(f"{name} 必须 >= 0，当前: {val}")
        if not 0.0 < th_es_entry <= 1.0:
            raise ValueError(f"th_es_entry 必须在 (0, 1] 区间，当前: {th_es_entry}")
        if not (th_xs_crash < th_xs_exit < th_xs_reduce_high):
            raise ValueError(
                f"th_xs_crash 必须小于 th_xs_exit 且 th_xs_exit 必须小于 "
                f"th_xs_reduce_high，当前: "
                f"{th_xs_crash} / {th_xs_exit} / {th_xs_reduce_high}")
        if not 0.0 < base_decay_rate <= 1.0:
            raise ValueError(f"base_decay_rate 必须在 (0, 1] 区间，当前: {base_decay_rate}")
        if win_decay_grace < 0:
            raise ValueError(f"win_decay_grace 不能为负，当前: {win_decay_grace}")
        if pnl_decay_profit_mult <= 0 or pnl_decay_loss_mult <= 0:
            raise ValueError(
                f"衰减倍率必须为正，当前 profit: {pnl_decay_profit_mult}, "
                f"loss: {pnl_decay_loss_mult}")
        # 防 factor = 1-(1-base)*mult <= 0 → 非整数幂得 NaN 并沿 PS 传播
        if (1.0 - base_decay_rate) * max(pnl_decay_profit_mult,
                                         pnl_decay_loss_mult) >= 1.0:
            raise ValueError(
                f"(1-base_decay_rate)*mult 必须 < 1（防 factor<=0 → NaN），"
                f"当前 base={base_decay_rate}, profit={pnl_decay_profit_mult}, "
                f"loss={pnl_decay_loss_mult}")
        if mrs_clip <= 0:
            raise ValueError(f"mrs_clip 必须为正（有界化分母防除零），当前: {mrs_clip}")
        if time_decay_interval <= 0:
            raise ValueError(f"time_decay_interval 必须为正（指数分母防除零），"
                             f"当前: {time_decay_interval}")
        if th_reversal_gap >= 0:
            raise ValueError(f"th_reversal_gap 必须为负（低开跌幅），当前: {th_reversal_gap}")
        if th_reversal_ofss < 0:
            raise ValueError(f"th_reversal_ofss 不能为负，当前: {th_reversal_ofss}")
        if reversal_add_mult <= 0:
            raise ValueError(f"reversal_add_mult 必须为正，当前: {reversal_add_mult}")
        try:
            pd.Timestamp(reversal_window_end).time()
        except Exception:
            raise ValueError(
                f"reversal_window_end 必须为 HH:MM 时间串，当前: {reversal_window_end}")
        if not 0.0 < fund_stability_penalty <= 1.0:
            raise ValueError(f"fund_stability_penalty 必须在 (0, 1] 区间，"
                             f"当前: {fund_stability_penalty}")
        if not 0.0 < reduce_fraction <= 1.0:
            raise ValueError(f"reduce_fraction 必须在 (0, 1] 区间，当前: {reduce_fraction}")
        if es_sigmoid_k <= 0:
            raise ValueError(f"es_sigmoid_k 必须为正，当前: {es_sigmoid_k}")
        if not 0.0 < max_single_position <= 1.0:
            raise ValueError(f"max_single_position 必须在 (0, 1] 区间，当前: {max_single_position}")
        if not 0.0 < base_weight <= max_single_position:
            raise ValueError(f"base_weight 必须在 (0, max_single_position] 区间，"
                             f"当前: {base_weight} / 上限 {max_single_position}")
        if not 0.0 < reduce_step_ratio <= 1.0:
            raise ValueError(f"reduce_step_ratio 必须在 (0, 1] 区间，当前: {reduce_step_ratio}")
        for name, (lo, hi) in (("tw_gmod_clip", tw_gmod_clip),
                               ("tw_cmod_clip", tw_cmod_clip)):
            if lo <= 0 or hi < lo:
                raise ValueError(f"{name} 区间非法: [{lo}, {hi}]")

        # 旧参数名兼容：youzi_chase_th → th_retail_chase
        if youzi_chase_th is not None:
            th_retail_chase = youzi_chase_th
        if not 0.0 < th_retail_chase < 1.0:
            raise ValueError(f"th_retail_chase 必须在 (0, 1) 区间，当前: {th_retail_chase}")

        self.w_ofss, self.w_cps, self.w_inst, self.w_north = map(float, weights)
        self.chase_window = chase_window
        self.rs_window = rs_window
        self.industry_window = industry_window
        self.inst_window = inst_window
        self.th_retail_chase = th_retail_chase

        # A 股硬过滤层
        self.th_amount = th_amount
        self.start_time = pd.Timestamp(start_time).time()
        self.end_time = pd.Timestamp(end_time).time()

        # ES
        self.w_es_ms = w_es_ms
        self.w_es_purity = w_es_purity
        self.w_es_mrs = w_es_mrs
        self.mrs_clip = mrs_clip
        self.es_sigmoid_k = es_sigmoid_k
        self.th_es_entry = th_es_entry

        # PS
        self.base_decay_rate = base_decay_rate
        self.time_decay_interval = time_decay_interval
        self.win_decay_grace = win_decay_grace
        self.pnl_decay_profit_mult = pnl_decay_profit_mult
        self.pnl_decay_loss_mult = pnl_decay_loss_mult
        self.cancel_ratio_th = cancel_ratio_th
        self.fund_stability_penalty = fund_stability_penalty
        self.obi_thin_th = obi_thin_th
        self.big_thin_th = big_thin_th

        # XS
        self.w_xs_ms = w_xs_ms
        self.w_xs_purity = w_xs_purity
        self.w_xs_drawdown = w_xs_drawdown
        self.th_xs_exit = th_xs_exit
        self.th_xs_reduce_high = th_xs_reduce_high
        self.th_xs_crash = th_xs_crash
        # 反包
        self.th_reversal_gap = th_reversal_gap
        self.th_reversal_ofss = th_reversal_ofss
        self.reversal_add_mult = reversal_add_mult
        self.reversal_window_end = pd.Timestamp(reversal_window_end).time()
        self.reduce_fraction = reduce_fraction
        self.circuit_index_drop = circuit_index_drop

        # Target_Weight
        self.base_weight = base_weight
        self.max_single_position = max_single_position
        self.reduce_step_ratio = reduce_step_ratio
        self.tw_gmod_clip = (float(tw_gmod_clip[0]), float(tw_gmod_clip[1]))
        self.tw_cmod_clip = (float(tw_cmod_clip[0]), float(tw_cmod_clip[1]))

        # 兼容参数（保留构造，旧闸门语义已废弃）
        self.th_global_min = th_global_min
        self.th_adr_min = th_adr_min
        self.th_mrs_min = th_mrs_min
        self.th_industry_min = th_industry_min
        self.th_ms_bull = th_ms_bull
        self.th_lock = th_lock
        self.th_chase = th_chase
        self.th_purity = th_purity
        self.th_ms_exit = th_ms_exit
        self.th_slippage = th_slippage
        self.win_hold_max = win_hold_max
        self.th_grs_circuit = th_grs_circuit
        self.symbol_to_industry = dict(symbol_to_industry or {})
        self.rank_gate = rank_gate

    # ------------------------------------------------------------------
    # 信号合成（列级）
    # ------------------------------------------------------------------

    def _agent_ms(self, features: pd.DataFrame) -> pd.Series:
        """主体情绪分：加权合成，缺失分量按剩余权重重归一化，全缺失 → NaN。

        Inst_Flow 经无状态 _norm_flow 平滑（tanh 有界，量纲自适应成交额或固定
        1e7 基准，无 rolling_std / 打开盘前 N 根盲区）：
            norm_inst = tanh(inst_flow / scale)
        inst_flow 缺失 → norm 为 NaN（作为缺失分量参与权重重归一化，不强行置 0）。
        """
        amount = features["amount"] if "amount" in features.columns else None
        inst = _norm_flow(features["inst_flow"], amount)

        parts = pd.DataFrame({
            "ofss": features["ofss"].astype(float),
            "cps": features["cps"].astype(float),
            "inst": inst,
            "north": features["north_sync"].astype(float),
        })
        w = np.array([self.w_ofss, self.w_cps, self.w_inst, self.w_north])
        mask = parts.notna().to_numpy()
        wsum = (mask * w).sum(axis=1)
        score = np.where(mask, parts.fillna(0.0).to_numpy() * w, 0.0).sum(axis=1)
        # 全缺失时 wsum==0 → 保持 NaN
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

        返回长表 [ts, symbol, rs]；缺数据 → rs 全 NaN（评分中性处理，不阻断）。
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
        """最终市场情绪分（多空对称修正，量纲有界）。

        base_ms  = clip(agent_ms + chain_mod, -1, 1)；宏观作乘性门控：
            base_ms >= 0（多头偏向）→ scale = 1 + global_mod
            base_ms <  0（空头偏向）→ scale = 1 - global_mod
                 （全球越恶化 global_mod 越负，空头信号被放大而非压制）
        final_ms = clip(base_ms * scale, -1, 1)。
        Chain_Mod / Global_Mod 缺失视为中性 0；agent_ms 缺失 → 输出保持 NaN。
        """
        agent = agent_ms.astype(float)
        chain = chain_mod.fillna(0.0).astype(float).to_numpy()
        gmod = global_mod.fillna(0.0).astype(float).to_numpy()

        base = np.clip(agent.fillna(0.0).to_numpy() + chain, -1.0, 1.0)
        scale = np.where(base >= 0.0, 1.0 + gmod, 1.0 - gmod)
        out = np.clip(base * scale, -1.0, 1.0)
        # agent 缺失 → 恢复 NaN（不能让 fillna(0) 的中间计算污染输出）
        out = np.where(agent.isna().to_numpy(), np.nan, out)
        return pd.Series(out, index=agent_ms.index)

    def capital_purity(self, features: pd.DataFrame) -> pd.Series:
        """资金纯净度：聪明钱增益 - 噪声成分惩罚（连续化，动态缺失重归一化）。

        资金流全部经无状态 _norm_flow 平滑（tanh 有界，量纲自适应成交额或固定
        1e7 基准，无 rolling_std / 打开盘前 N 根盲区 / 反转信号滞后）。

        固定权重：w_inst=0.35, w_north=0.25, w_retail=0.20, w_youzi=0.20。
        分子 = w_inst*norm_inst + w_north*north
               - w_retail*norm_retail - w_youzi*norm_youzi
        分母 = 仅对非 NaN 分量的权重求和 w_sum（与 Agent_MS 完全一致的动态
        权重重归一化）：全缺失 → w_sum==0 → 输出 NaN；
        否则 purity = clip(numerator / w_sum, -1, 1)。
        north_sync 缺失 → 视为该分量缺失（不再 fillna(0) 参与）。
        """
        amount = features["amount"] if "amount" in features.columns else None
        norm_inst = _norm_flow(features["inst_flow"], amount)
        norm_retail = _norm_flow(features["retail_flow"], amount)
        norm_youzi = _norm_flow(features["youzi_flow"], amount)
        north = features["north_sync"].astype(float).to_numpy()

        w_inst, w_north, w_retail, w_youzi = 0.35, 0.25, 0.20, 0.20

        m_inst = features["inst_flow"].notna().to_numpy()
        m_north = features["north_sync"].notna().to_numpy()
        m_retail = features["retail_flow"].notna().to_numpy()
        m_youzi = features["youzi_flow"].notna().to_numpy()

        contrib = (
            np.where(m_inst, w_inst * norm_inst, 0.0)
            + np.where(m_north, w_north * north, 0.0)
            - np.where(m_retail, w_retail * norm_retail, 0.0)
            - np.where(m_youzi, w_youzi * norm_youzi, 0.0)
        )
        w_sum = (m_inst.astype(float) * w_inst + m_north.astype(float) * w_north
                 + m_retail.astype(float) * w_retail
                 + m_youzi.astype(float) * w_youzi)
        out = np.where(w_sum > 0, contrib / np.where(w_sum > 0, w_sum, np.nan),
                       np.nan)
        return pd.Series(np.clip(out, -1.0, 1.0), index=features.index)

    # ------------------------------------------------------------------
    # 状态判定（无状态纯函数）
    # ------------------------------------------------------------------

    def state_of(self, row: pd.Series) -> str:
        """逐行判定状态机状态（S_youzi_only 优先级最高，其次 S_push，默认 S_noise）。"""
        youzi_only = (row["youzi_flow"] > 0) and (row["inst_flow"] <= 0) \
            and (row["retail_chase"] > self.th_retail_chase)
        if youzi_only:
            return S_YOUZI_ONLY
        push = (row["final_ms"] > 0) and (row["ofss"] > 0) \
            and (row["inst_flow"] > 0) and (row["cps"] > 0)
        if push:
            return S_PUSH
        return S_NOISE

    # ------------------------------------------------------------------
    # 连续评分：入场分 ES
    # ------------------------------------------------------------------

    @staticmethod
    def _as_series(x, dtype, fill, index=None) -> pd.Series:
        """统一为 pd.Series（缺失 → fill 值），依托 index 自动对齐参与运算。

        兼容标量 / 长度为 1 的 Series；标量 NaN 也按 fill 兜底。提供 `index` 时，
        标量按该索引广播（避免与等长 Series 混乘时因 outer-join 产生 NaN）。
        """
        if isinstance(x, pd.Series):
            return x.astype(dtype).fillna(fill)
        if index is not None:
            return pd.Series(np.broadcast_to(x, len(index)), index=index,
                             dtype=dtype).fillna(fill)
        if pd.isna(x):
            return pd.Series([fill], dtype=dtype)
        return pd.Series([x], dtype=dtype)

    def _compute_es(self, final_ms, capital_purity, mrs,
                    s_youzi_only) -> pd.Series:
        """入场分 ES ∈ (0, 1)：批量向量化，长表整列计算。

        全流程以 pd.Series 参与加减乘除，依托各自 index 自动对齐（不做 numpy
        位置剥离）：
            final_ms       已严格在 [-1, 1]，直接使用，缺失 → 0
            capital_purity 已严格在 [-1, 1]，直接使用，缺失 → 0
            mrs            值域未定，做无量纲有界缩放，缺失 → 0
            s_youzi_only   布尔型，标记游资独舞

        es_raw  = w_es_ms*final_ms + w_es_purity*capital_purity
                  + w_es_mrs*mrs_c
        mrs_c   = clip(mrs, ±mrs_clip) / mrs_clip
        es_score = sigmoid(es_sigmoid_k * es_raw)，有限输入严格位于 (0, 1)
        游资独舞 s_youzi_only → 硬掩码否决：es_final = es_score.mask(youzi, 0.0)
        直接输出 0.0（不做数学衰减）。
        """
        fms = self._as_series(final_ms, float, 0.0)
        purity = self._as_series(capital_purity, float, 0.0)
        mrs_s = self._as_series(mrs, float, 0.0)
        youzi = self._as_series(s_youzi_only, bool, False)

        mrs_c = mrs_s.clip(-self.mrs_clip, self.mrs_clip) / self.mrs_clip

        es_raw = (self.w_es_ms * fms + self.w_es_purity * purity
                  + self.w_es_mrs * mrs_c)
        es_score = 1.0 / (1.0 + np.exp(-self.es_sigmoid_k * es_raw))
        es_final = es_score.mask(youzi, 0.0)
        return es_final.rename("es")

    def calculate_entry_score(self, row: pd.Series) -> float:
        """入场分 ES ∈ (0, 1)：单行包装，公式与 _compute_es 完全一致。

        ES_raw = w_es_ms*final_ms + w_es_purity*capital_purity + w_es_mrs*mrs_c
        ES     = sigmoid(es_sigmoid_k * ES_raw)
        游资独舞（state==S_YOUZI_ONLY）→ ES 硬掩码为 0.0。
        """
        youzi = bool(row.get("state", S_NOISE) == S_YOUZI_ONLY)
        es = self._compute_es(
            pd.Series([row.get("final_ms", np.nan)]),
            pd.Series([row.get("capital_purity", np.nan)]),
            pd.Series([row.get("mrs", np.nan)]),
            pd.Series([youzi]),
        )
        return float(es.iloc[0])

    # ------------------------------------------------------------------
    # 连续评分：持仓分 PS
    # ------------------------------------------------------------------

    def _compute_ps(self, es, time_decay, fund_stability) -> pd.Series:
        """持仓分 PS ∈ [0, 1]：批量向量化，长表整列计算。

        全流程以 pd.Series 参与连乘，依托 index 自动对齐（不做 numpy 位置剥离）；
        兼容标量 / 长度为 1 的 Series：
            es             ∈ [0, 1)，缺失 → 0
            time_decay     ∈ [0.1, 1.0]，缺失 → 1.0（不打折）
            fund_stability ∈ {penalty, 1.0}，缺失 → 1.0（不打折）

        ps_score = es * time_decay * fund_stability，各分量天然有界 → 连乘恒在
        [0, 1]，不再 clip。
        """
        idx = next((s.index for s in (es, time_decay, fund_stability)
                    if isinstance(s, pd.Series)), None)
        es_s = self._as_series(es, float, 0.0, idx)
        td_s = self._as_series(time_decay, float, 1.0, idx)
        fs_s = self._as_series(fund_stability, float, 1.0, idx)
        return (es_s * td_s * fs_s).rename("ps")

    def calculate_position_score(self, row: pd.Series, pos: Position) -> float:
        """持仓分 PS ∈ [0, 1] = ES * Time_Decay * Fund_Stability（标量包装）。"""
        ps = self._compute_ps(self.calculate_entry_score(row),
                              self.time_decay(row, pos),
                              row.get("fund_stability", 1.0))
        return float(ps.iloc[0])

    def time_decay(self, row: pd.Series, pos: Position) -> float:
        """时间衰减（衰减保护期 + 浮盈亏非对称衰减）。纯数学安全 / 无冗余分支。

        - effective = max(0.0, bars_held - win_decay_grace)；effective=0 时次幂为
          1.0，直接省去旧代码"保护期内 return 1.0"的冗余分支。
        - pnl_ratio = (close - avg_cost) / avg_cost；防除零与空值：avg_cost<=1e-6
          或含 NaN → 缺失 → raw_factor = base_decay_rate（中性）。
        - 浮盈(pnl>0) / 浮亏(pnl<=0) 分别套用 profit / loss mult 得 raw_factor。
        - factor = max(0.01, raw_factor)（纵深防御，严禁负/零底数 → 防复数/NaN）。
        - td = factor ** (effective / time_decay_interval)，clip 到 [0.1, 1.0]。
        """
        close = float(row["close"])
        avg_cost = float(pos.avg_cost)
        effective = max(0.0, float(pos.bars_held) - self.win_decay_grace)

        if avg_cost <= 1e-6 or np.isnan(close) or np.isnan(avg_cost):
            raw_factor = self.base_decay_rate
        else:
            pnl = (close - avg_cost) / avg_cost
            if pnl > 0:
                raw_factor = 1.0 - (1.0 - self.base_decay_rate) * self.pnl_decay_profit_mult
            else:
                raw_factor = 1.0 - (1.0 - self.base_decay_rate) * self.pnl_decay_loss_mult

        factor = max(0.01, raw_factor)
        td_val = factor ** (effective / self.time_decay_interval)
        return float(np.clip(td_val, 0.1, 1.0))

    def _compute_fund_stability(self, cancel_ratio_series, obi_series,
                                big_flow_series) -> pd.Series:
        """资金稳定性系数（纯向量化，长表整列预计算，状态机仅消费结果列）。

        is_cancel    = cancel.fillna(0.0)  > cancel_ratio_th   （NaN 视为不撤单）
        is_thin_obi  = |obi.fillna(1.0)|   < obi_thin_th       （NaN 视为不薄）
        is_thin_flow = |big_flow.fillna(1.0)| < big_thin_th
        mask = is_cancel | (is_thin_obi & is_thin_flow)
        命中 mask → fund_stability_penalty，否则 1.0。输出与输入 index 对齐。
        """
        cancel = cancel_ratio_series.fillna(0.0)
        is_cancel = cancel > self.cancel_ratio_th
        is_thin_obi = np.abs(obi_series.fillna(1.0)) < self.obi_thin_th
        is_thin_flow = np.abs(big_flow_series.fillna(1.0)) < self.big_thin_th
        is_thin = is_thin_obi & is_thin_flow
        mask = is_cancel | is_thin
        fs = pd.Series(1.0, index=mask.index)
        fs.loc[mask] = self.fund_stability_penalty
        return fs

    # ------------------------------------------------------------------
    # 连续评分：出局分 XS
    # ------------------------------------------------------------------

    def calculate_exit_score(self, row: pd.Series, pos: Position) -> float:
        """出局分 XS ∈ [-1, 1]（业务适配器：提取标量 + 调纯函数）。

        XS = w_xs_ms*Final_MS + w_xs_purity*Capital_Purity
             - w_xs_drawdown*Drawdown_From_High
        Final_MS / Capital_Purity 已在 [-1, 1]，直接使用原值（缺失 → 0），
        不再二次 clip 缩放。一票否决（游资溃逃 / 大盘跳水）→ 强制 XS = -1.0。
        所有数学计算委托给纯函数 _compute_xs。
        """
        ms = float(row.get("final_ms", 0.0))
        purity = float(row.get("capital_purity", 0.0))
        close = float(row.get("close", 0.0))
        hwm = float(pos.high_price_watermark)
        is_veto = bool(self.veto(row))
        ws = (self.w_xs_ms, self.w_xs_purity, self.w_xs_drawdown)
        return self._compute_xs(ms, purity, hwm, close, is_veto, ws)

    @staticmethod
    def _compute_xs(final_ms, capital_purity, high_price_watermark,
                    close, veto_flag, weights) -> float:
        """出局分 XS ∈ [-1, 1]（纯数学算子，无业务/状态依赖）。

        NaN/None → 0（空值中性，中立不提供出局信号）。
        回撤 dd = max(0, (hwm - close) / hwm)；hwm<=1e-6 → 0（防除零）。
        veto_flag=True → 一票否决，无视权重直接返回 -1.0。
        xs_raw = w_ms*final_ms + w_purity*capital_purity - w_dd*dd，clip 到 [-1, 1]。
        """
        w_ms, w_purity, w_dd = weights
        ms = float(final_ms) if not (final_ms is None or pd.isna(final_ms)) else 0.0
        purity = float(capital_purity) if not (capital_purity is None
                                               or pd.isna(capital_purity)) else 0.0
        if veto_flag:
            return -1.0
        if high_price_watermark <= 1e-6:
            dd = 0.0
        else:
            dd = max(0.0, (high_price_watermark - close) / high_price_watermark)
        xs_raw = w_ms * ms + w_purity * purity - w_dd * dd
        return float(np.clip(xs_raw, -1.0, 1.0))

    def drawdown_from_high(self, row: pd.Series, pos: Position) -> float:
        """持仓回撤（相对持仓期最高价）：max(0, (HWM - close) / HWM)。"""
        close = row.get("close", np.nan)
        hwm = pos.high_price_watermark
        if pd.isna(close) or hwm <= 0:
            return 0.0
        return max(0.0, (hwm - float(close)) / hwm)

    def veto(self, row: pd.Series) -> bool:
        """一票否决（强制 XS = -1.0）：

        ① 游资溃逃：big_flow < 0（大资金净流出）且 Retail_Chase > th_retail_chase
        ② 大盘跳水：沪深300 日内跌破 VWAP * (1 - circuit_index_drop)
        任一输入缺失 → 对应条件不触发（保守不误杀）。
        """
        big = row.get("big_flow", np.nan)
        chase = row.get("retail_chase", np.nan)
        run_away = (pd.notna(big) and float(big) < 0.0
                    and pd.notna(chase) and float(chase) > self.th_retail_chase)
        idx_close = row.get("index_close", np.nan)
        idx_vwap = row.get("index_vwap", np.nan)
        idx_dive = (pd.notna(idx_close) and pd.notna(idx_vwap)
                    and float(idx_close) < float(idx_vwap) * (1.0 - self.circuit_index_drop))
        return bool(run_away or idx_dive)

    # ------------------------------------------------------------------
    # A 股硬过滤层（一票否决前置条件，独立于连续评分）
    # ------------------------------------------------------------------

    def hard_filters(self, row: pd.Series) -> Dict[str, bool]:
        """A 股硬约束明细（全部为真才允许开仓；NaN 比较恒为 False 保守关闭）。"""
        return {
            "st": not bool(row.get("is_st", True)),
            "limit": bool(self._gt(row, "up_limit", row.get("close", np.nan))
                          and self._gt(row, "close", row.get("down_limit", np.nan))),
            "liquidity": self._gt(row, "amount", self.th_amount),
            "time": self._in_time_window(row),
        }

    def hard_all(self, row: pd.Series) -> bool:
        """A 股硬过滤层：全部通过才允许开仓。"""
        return all(self.hard_filters(row).values())

    # ------------------------------------------------------------------
    # 目标持仓比例 Target_Weight（撮合引擎差额调仓的目标输入）
    # ------------------------------------------------------------------

    def _tw_scale(self, row: pd.Series) -> float:
        """Target_Weight 乘子 = clip(1+Global_Mod) × clip(1+Chain_Mod)。

        Global_Mod / Chain_Mod 缺失视为中性 0（乘子 = 1.0，不放大不缩小）。
        """
        g = row.get("global_mod", np.nan)
        c = row.get("chain_mod", np.nan)
        gmod = 0.0 if pd.isna(g) else float(g)
        cmod = 0.0 if pd.isna(c) else float(c)
        gs = float(np.clip(1.0 + gmod, *self.tw_gmod_clip))
        cs = float(np.clip(1.0 + cmod, *self.tw_cmod_clip))
        return gs * cs

    def generate_target_weights(self, row: pd.Series,
                                pos: Optional[Position]) -> float:
        """目标持仓比例 Target_Weight ∈ [0, max_single_position]。

        未持仓（pos is None）：
            A 股硬过滤全过 且 ES >= th_es_entry
                → base_weight * ES * clip(1+Global_Mod) * clip(1+Chain_Mod)
            否则 → 0.0
        持仓（XS 四分判定链）：
            XS >= th_xs_reduce_high      → base_weight * PS * 乘子（正常持仓按 PS 调仓）
            th_xs_exit < XS < th_xs_reduce_high
                                         → simulated_weight * reduce_step_ratio
                                            （容错阶梯减仓，不清仓）
            th_xs_crash < XS <= th_xs_exit → 0.0（常规清仓）
            XS <= th_xs_crash 或一票否决   → 0.0（极速清仓 Crash / Panic Exit）
        统一 clip(0, max_single_position) 兜底。
        """
        if pos is None:
            es = self.calculate_entry_score(row)
            if self.hard_all(row) and es >= self.th_es_entry:
                tw = self.base_weight * es * self._tw_scale(row)
            else:
                tw = 0.0
        else:
            xs = self.calculate_exit_score(row, pos)
            if xs <= self.th_xs_crash or self.veto(row):
                tw = 0.0  # 极速清仓
            elif xs <= self.th_xs_exit:
                tw = 0.0  # 常规清仓
            elif xs < self.th_xs_reduce_high:
                tw = pos.simulated_weight * self.reduce_step_ratio
            else:
                ps = self.calculate_position_score(row, pos)
                tw = self.base_weight * ps * self._tw_scale(row)
        return float(np.clip(tw, 0.0, self.max_single_position))

    # ------------------------------------------------------------------
    # 历史接口兼容包装（旧 8 层开仓闸门 / 6 层平仓闸门语义 → 新架构映射）
    # ------------------------------------------------------------------

    def entry_gates(self, row: pd.Series) -> Dict[str, bool]:
        """兼容接口：原 8 层开仓闸门明细。

        新语义：全球/系统/产业/Alpha/个股 5 层信号闸门统一由「ES >= th_es_entry」
        表达（信号强度连续化）；liquidity/time 由硬过滤层承担；rank 保持原样。
        全部为真 ≈ 可开仓（与 TradingStateMachine 新决策一致）。
        """
        es = self.calculate_entry_score(row)
        hard = self.hard_filters(row)
        sig_ok = es >= self.th_es_entry
        return {
            "global": sig_ok,
            "system": sig_ok,
            "beta": sig_ok,
            "industry": sig_ok,
            "alpha": sig_ok,
            "stock": sig_ok,
            "rank": self.rank_gate.passes(row) if self.rank_gate else True,
            "liquidity": bool(hard["st"] and hard["limit"] and hard["liquidity"]),
            "time": hard["time"],
        }

    def entry_all(self, row: pd.Series) -> bool:
        """兼容接口：原「全部开仓闸门为真」。等价于 硬过滤全过 且 ES 达标。"""
        return all(self.entry_gates(row).values())

    def exit_triggers(self, row: pd.Series, pos: Position) -> Dict[str, bool]:
        """兼容接口：原 6 层平仓闸门明细。

        新语义：state/ms/purity/stop/hold_time 统一由「XS <= th_xs_exit」（清仓线）
        表达；crash/circuit 由极速清仓线（XS <= th_xs_crash）或一票否决表达。
        """
        xs = self.calculate_exit_score(row, pos)
        exit_ = xs <= self.th_xs_exit
        crash = bool(xs <= self.th_xs_crash or self.veto(row))
        return {
            "state": exit_,
            "ms": exit_,
            "purity": exit_,
            "stop": exit_,
            "hold_time": exit_,
            "crash": crash,
            "circuit": bool(exit_ or crash),
        }

    def exit_any(self, row: pd.Series, pos: Position) -> bool:
        """兼容接口：原「任一平仓闸门为真」→ 清仓。"""
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
        """当前时间在 [start_time, end_time] 内。"""
        t = row.get("ts", row.name)
        try:
            cur = pd.Timestamp(t).time()
        except Exception:
            return False
        return self.start_time <= cur <= self.end_time

    # ------------------------------------------------------------------
    # 次日低开反包（Reversal / Counter-Attack）
    # ------------------------------------------------------------------

    def reversal_active(self, row: pd.Series, pos: "Position") -> bool:
        """反包激活判定（ALL MUST BE TRUE）：

        ① 已有持仓且跨入次日（T >= T+1）
        ② 时间窗 [start_time, reversal_window_end]——受开仓 time 闸门共同约束，
           默认 start_time=10:00 时反包窗口仅剩 10:00 整（禁买时段不触发）
        ③ 深度低开/下探：(Current - Prev_Close) / Prev_Close <= th_reversal_gap
        ④ 盘口承接：OFSS > th_reversal_ofss
        ⑤ 大资金逆势承接：Capital_Purity > 0 且 big_flow > 0
        任一输入缺失 → 对应条件不触发（保守不误判）。
        """
        if pos is None:
            return False
        cur_ts = row.get("ts", row.name)
        if cur_ts is None:
            return False
        cur = pd.Timestamp(cur_ts)
        # ① 跨入次日
        if cur.date() <= pd.Timestamp(pos.entry_time).date():
            return False
        # ② 时间窗（含开仓 time 闸门约束）
        cur_t = cur.time()
        if not (self.start_time <= cur_t <= self.reversal_window_end):
            return False
        # ③ 深度低开/下探
        prev = row.get("prev_close", np.nan)
        close = row.get("close", np.nan)
        gap_ok = (pd.notna(prev) and float(prev) > 0 and pd.notna(close)
                  and (float(close) - float(prev)) / float(prev) <= self.th_reversal_gap)
        # ④ 盘口承接分
        ofss = row.get("ofss", np.nan)
        ofss_ok = bool(pd.notna(ofss) and float(ofss) > self.th_reversal_ofss)
        # ⑤ 大资金逆势承接
        purity = row.get("capital_purity", np.nan)
        big = row.get("big_flow", np.nan)
        money_ok = bool(pd.notna(purity) and float(purity) > 0.0
                        and pd.notna(big) and float(big) > 0.0)
        return bool(gap_ok and ofss_ok and money_ok)

    def reversal_overridden(self, row: pd.Series) -> bool:
        """反包保护立即失效（强制清仓）：
        ① 大盘熔断（沪深300 跌破 VWAP*(1-circuit_index_drop)）
        ② 游资出逃：big_flow < -0.5
        """
        big = row.get("big_flow", np.nan)
        flight = bool(pd.notna(big) and float(big) < -0.5)
        idx_close = row.get("index_close", np.nan)
        idx_vwap = row.get("index_vwap", np.nan)
        dive = bool(pd.notna(idx_close) and pd.notna(idx_vwap)
                    and float(idx_close)
                    < float(idx_vwap) * (1.0 - self.circuit_index_drop))
        return bool(flight or dive)


class TradingStateMachine:
    """连续评分 + A 股硬过滤的交易状态机（有状态）。

    逐 Bar / 逐标的推进：
    - 状态判定（S_push / S_youzi_only / S_noise）由 SignalSynthesizer.state_of 完成
    - 开仓：未持仓 且 A 股硬过滤全过 且 ES >= th_es_entry → BUY
    - 平仓：XS <= th_xs_exit（常规清仓）或 XS <= th_xs_crash / 一票否决
            （极速清仓 Crash / Panic Exit）→ SELL
    - 容错阶梯减仓：th_xs_exit < XS < th_xs_reduce_high 且 满足最小减仓间隔
            → DECAY_REDUCE
    - 加仓：已持仓 且 XS >= th_xs_reduce_high 且 S_push 且（可选）全部开仓闸门
            且 满足最小加仓间隔 → ADD
    - 其余 → HOLD（每根 Bar / 每标的均输出一个 Signal）

    内部维护 paper positions（入场时间 / 入场 VWAP / 加权成本 / 持仓最高价 /
    持仓分钟数），供持仓评分使用；连续回测可跨 run() 调用保留，也可 reset()。
    """

    def __init__(
        self,
        synthesizer: Optional[SignalSynthesizer] = None,
        min_add_interval: int = 5,   # 两次加仓之间的最小 Bar 数
        add_requires_entry_gates: bool = True,  # 加仓是否复用全部开仓闸门
        min_reduce_interval: int = 5,  # 两次阶梯减仓之间的最小 Bar 数
    ) -> None:
        self.syn = synthesizer or SignalSynthesizer()
        if min_add_interval < 0:
            raise ValueError(f"min_add_interval 不能为负，当前: {min_add_interval}")
        if min_reduce_interval < 0:
            raise ValueError(f"min_reduce_interval 不能为负，当前: {min_reduce_interval}")
        self.min_add_interval = min_add_interval
        self.add_requires_entry_gates = add_requires_entry_gates
        self.min_reduce_interval = min_reduce_interval
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
        # 横截面排序闸门开启时，追加因子 T-1 全池排名列（当日每 bar 复用昨日排名）
        if self.syn.rank_gate is not None:
            ev = add_previous_rank_columns(
                ev.reset_index().rename(columns={"index": "ts"}),
                [self.syn.rank_gate.factor]).set_index("ts")
        ev = ev.reset_index().rename(columns={"index": "ts"})

        # 个股合规 / 价格列（硬过滤与持仓评分）
        k = ds.kline.copy().reset_index().rename(columns={"index": "ts"})
        k_cols = ["ts", SYMBOL, "close", "vwap", "amount",
                  "up_limit", "down_limit", "is_st"]
        k = k[[c for c in k_cols if c in k.columns]]
        ev = ev.merge(k, on=["ts", SYMBOL], how="left")

        # 昨收 prev_close（反包 gap 判定）：override 特征列优先；否则按 symbol
        # 每日最后收盘价 shift 补齐（每周期首日 NaN，反包要求跨日天然规避；
        # 只用历史收盘，防未来安全）
        if "prev_close" not in ev.columns:
            ev = ev.copy()
            ev["date"] = ev["ts"].dt.normalize()
            kd = k[["ts", SYMBOL, "close"]].copy()
            kd["date"] = kd["ts"].dt.normalize()
            day_close = kd.groupby([SYMBOL, "date"])["close"].last().reset_index()
            day_close["prev_close"] = day_close.groupby(SYMBOL)["close"].shift(1)
            ev = ev.merge(day_close[[SYMBOL, "date", "prev_close"]],
                          on=[SYMBOL, "date"], how="left")
            ev = ev.drop(columns=["date"])

        # 市场列（指数 VWAP 熔断）：取首个指数代码
        if ds.index_min is not None and not ds.index_min.empty:
            idx = ds.index_min.copy()
            code = idx["index_code"].iloc[0]
            idx = idx[idx["index_code"] == code].reset_index().rename(
                columns={"index": "ts", "close": "index_close",
                         "vwap": "index_vwap"})
            ev = ev.merge(idx[["ts", "index_close", "index_vwap", "ma20", "ma60"]],
                          on="ts", how="left")

        # 状态列（逐行无状态判定）
        ev["state"] = ev.apply(self.syn.state_of, axis=1)
        # 资金稳定性系数：纯向量化预计算整列，状态机仅消费本列（缺列 → 全 NaN → 1.0）
        s_cancel = ev.get("cancel_ratio", pd.Series(np.nan, index=ev.index))
        s_obi = ev.get("obi", pd.Series(np.nan, index=ev.index))
        s_flow = ev.get("big_flow", pd.Series(np.nan, index=ev.index))
        ev["fund_stability"] = self.syn._compute_fund_stability(s_cancel, s_obi, s_flow)
        assert ev["ts"].is_monotonic_increasing, "评估表必须按时间升序"
        return ev

    # ------------------------------------------------------------------
    # 内部：逐行决策
    # ------------------------------------------------------------------

    def _scores(self, row: pd.Series, pos: Optional[Position]) -> Dict[str, object]:
        """评分快照（ES/PS/XS/Target_Weight 及分量），写入 Signal.metrics 供复盘与撮合。"""
        scores: Dict[str, object] = {
            "es": self.syn.calculate_entry_score(row),
            "target_weight": self.syn.generate_target_weights(row, pos),
        }
        if pos is not None:
            scores["ps"] = self.syn.calculate_position_score(row, pos)
            scores["xs"] = self.syn.calculate_exit_score(row, pos)
            scores["xs_crash"] = bool(
                scores["xs"] <= self.syn.th_xs_crash or self.syn.veto(row))
            scores["time_decay"] = self.syn.time_decay(row, pos)
            scores["fund_stability"] = row.get("fund_stability", 1.0)
            scores["drawdown"] = self.syn.drawdown_from_high(row, pos)
            scores["reduce_fraction"] = self.syn.reduce_fraction
            scores["reduce_step_ratio"] = self.syn.reduce_step_ratio  # 供撮合 fallback 统一比例
        return scores

    def _on_flat(self, row: pd.Series) -> Signal:
        """无持仓：A 股硬过滤全过 且 ES >= th_es_entry → BUY，否则 HOLD。"""
        sym = row[SYMBOL]
        gates = self.syn.entry_gates(row)
        scores = self._scores(row, None)  # 含 target_weight（开仓目标权重）
        if self.syn.hard_all(row) and self.syn.calculate_entry_score(row) \
                >= self.syn.th_es_entry:
            close = float(row["close"])
            vwap = float(row["vwap"])
            self.positions[sym] = Position(
                symbol=sym, entry_time=pd.Timestamp(row["ts"]),
                entry_vwap=vwap, last_price=close, bars_held=0, last_add_bar=0,
                last_reduce_bar=0, high_price_watermark=close, avg_cost=vwap,
                simulated_weight=float(scores["target_weight"]))
            return Signal(sym, pd.Timestamp(row["ts"]), ACT_BUY,
                          row["state"],
                          self._metrics(row, entry_gates=gates, scores=scores))
        return Signal(sym, pd.Timestamp(row["ts"]), ACT_HOLD,
                      row["state"],
                      self._metrics(row, entry_gates=gates, scores=scores))

    @staticmethod
    def _rebase_avg_cost(row: pd.Series, pos: Position,
                         sim_old: float, target: float) -> None:
        """加仓后按市价重算加权成本 avg_cost（对齐 Position 注释）。

        new = (old_cost × sim_old + 当前价 × Δ) / (sim_old + Δ)，Δ = target - sim_old。
        close 缺失/非法或未加仓（Δ<=0）时保持旧成本（保守不恶化基准）。
        """
        price = row.get("close", np.nan)
        delta = target - sim_old
        if (pd.notna(price) and float(price) > 0 and delta > 0
                and sim_old > 0 and pos.avg_cost > 0):
            pos.avg_cost = ((pos.avg_cost * sim_old + float(price) * delta)
                            / (sim_old + delta))

    def _on_holding(self, row: pd.Series, pos: Position) -> Signal:
        """已持仓：先递增持仓分钟数与最高价，再按 XS / PS 决策。"""
        pos.bars_held += 1
        pos.last_price = float(row["close"])
        if float(row["close"]) > pos.high_price_watermark:
            pos.high_price_watermark = float(row["close"])

        xs = self.syn.calculate_exit_score(row, pos)
        scores = self._scores(row, pos)  # 含 target_weight（对应各 XS 分支的目标权重）

        # 次日低开反包：冻结 XS 清仓/阶梯减仓 + 承接加仓（受 time 闸门/窗口约束）
        if self.syn.reversal_active(row, pos) \
                and not self.syn.reversal_overridden(row):
            target = pos.simulated_weight
            added = False
            if pos.bars_held - pos.last_add_bar >= self.min_add_interval:
                es = self.syn.calculate_entry_score(row)
                sim_old = pos.simulated_weight
                target = min(sim_old
                             + self.syn.base_weight * es * self.syn.reversal_add_mult,
                             self.syn.max_single_position)
                self._rebase_avg_cost(row, pos, sim_old, target)
                pos.simulated_weight = target
                pos.last_add_bar = pos.bars_held
                scores["reversal_add"] = True
                added = True
            scores["target_weight"] = target
            return Signal(row[SYMBOL], pd.Timestamp(row["ts"]),
                          ACT_ADD if added else ACT_HOLD, row["state"],
                          self._metrics(row, scores=scores))

        # 清仓：常规清仓（XS <= th_xs_exit）或 极速清仓（XS <= th_xs_crash / 一票否决）
        if xs <= self.syn.th_xs_exit or self.syn.veto(row):
            del self.positions[row[SYMBOL]]
            return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_SELL,
                          row["state"],
                          self._metrics(row,
                                        exit_triggers=self.syn.exit_triggers(row, pos),
                                        scores=scores))

        # 容错阶梯减仓：XS 落入 (th_xs_exit, th_xs_reduce_high) 且满足减仓节奏
        if xs < self.syn.th_xs_reduce_high:
            if pos.bars_held - pos.last_reduce_bar >= self.min_reduce_interval:
                pos.last_reduce_bar = pos.bars_held
                pos.simulated_weight = float(scores["target_weight"])  # = 当前 × 0.8
                return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_DECAY_REDUCE,
                              row["state"], self._metrics(row, scores=scores))
            return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_HOLD,
                          row["state"], self._metrics(row, scores=scores))

        # 加仓：S_push 且（可选）全部开仓闸门通过 且 满足最小加仓间隔
        add_ok = (row["state"] == S_PUSH)
        if self.add_requires_entry_gates:
            add_ok = add_ok and self.syn.entry_all(row)
        if add_ok and (pos.bars_held - pos.last_add_bar >= self.min_add_interval):
            pos.last_add_bar = pos.bars_held
            sim_old = pos.simulated_weight
            target = float(scores["target_weight"])  # = base × PS × 乘子
            self._rebase_avg_cost(row, pos, sim_old, target)
            pos.simulated_weight = target
            return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_ADD,
                          row["state"], self._metrics(row, scores=scores))
        return Signal(row[SYMBOL], pd.Timestamp(row["ts"]), ACT_HOLD,
                      row["state"], self._metrics(row, scores=scores))

    @staticmethod
    def _metrics(row: pd.Series,
                 entry_gates: Optional[Dict[str, bool]] = None,
                 exit_triggers: Optional[Dict[str, bool]] = None,
                 scores: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """信号附带的指标快照（合成分 + 评分 + 触发的闸门明细）。"""
        m: Dict[str, object] = {}
        for col in ("agent_ms", "final_ms", "capital_purity", "retail_chase",
                    "rs", "industry_ms", "ofss", "cps", "inst_flow",
                    "north_sync", "lock_ratio", "mrs", "irs", "grs",
                    "global_mod", "chain_mod"):
            v = row.get(col, np.nan)
            m[col] = float(v) if pd.notna(v) else None
        if scores:
            for k, v in scores.items():
                m[k] = float(v) if v is not None else None
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
