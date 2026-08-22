"""策略默认超参数注册表（SignalSynthesizer 连续评分 / A 股硬过滤 / 兼容参数）。

所有键与 strategy/signals.SignalSynthesizer.__init__ 入参一一对应，
供 main.py 的 SMOKE_PARAMS / REAL_PARAMS 与 optimizer 寻优复用；
optuna 采样范围见 optimizer/search_space.py。
"""

# ---- 连续评分：入场分 ES ----
W_ES_MS = 0.4            # Final_MS 权重
W_ES_PURITY = 0.3        # Capital_Purity 权重
W_ES_MRS = 0.3           # MRS 权重
FINAL_MS_CLIP = 2.0      # Final_MS 有界化半宽（clip ±2 后 /2）
MRS_CLIP = 3.0           # MRS 有界化半宽（clip ±3 后 /3）
ES_SIGMOID_K = 3.0       # sigmoid 陡度
ES_YOUZI_DECAY = 0.5     # S_youzi_only 状态衰减系数
TH_ES_ENTRY = 0.4        # 开仓 ES 门槛

# ---- 连续评分：持仓分 PS ----
TIME_DECAY_BASE = 0.95       # 每 TIME_DECAY_INTERVAL 分钟衰减基数
TIME_DECAY_INTERVAL = 10.0   # 时间衰减步长（分钟）
MOMENTUM_EXEMPT = 0.015      # 浮盈安全垫（超此则时间衰减豁免）
CANCEL_RATIO_TH = 0.25       # 撤单率阈值（超过 → 资金不稳定）
FUND_STABILITY_PENALTY = 0.7 # 资金不稳定惩罚系数
OBI_THIN_TH = 0.05           # 盘口变薄：|OBI| 阈值（近似）
BIG_THIN_TH = 0.05           # 盘口变薄：|big_flow| 阈值（近似）

# ---- 连续评分：出局分 XS ----
W_XS_MS = 0.5            # Final_MS 权重
W_XS_PURITY = 0.3        # Capital_Purity 权重
W_XS_DRAWDOWN = 0.2      # 回撤权重
TH_XS_EXIT = 0.0         # XS <= 此值 → 清仓
TH_XS_REDUCE = 0.2       # XS < 此值 → 阶梯减仓
REDUCE_FRACTION = 0.5    # 阶梯减仓比例（每步卖出可卖份额的比例）
CIRCUIT_INDEX_DROP = 0.015  # 指数盘中跌破 VWAP 比例（1.5%）

# ---- 状态 / 一票否决共用 ----
TH_RETAIL_CHASE = 0.65   # S_youzi_only 与 XS 一票否决共用的追涨阈值

# ---- 目标权重 Target_Weight（驱动撮合引擎差额调仓）----
BASE_WEIGHT = 0.20              # 开仓/调仓基准权重（与 PositionSizer.base_position 同源）
MAX_SINGLE_POSITION = 0.30      # 目标权重单股上限（与 Account/PositionSizer 同值）
REDUCE_STEP_RATIO = 0.8         # 阶梯减仓目标比例（Target = simulated × 0.8）
TW_GMOD_CLIP = (0.2, 1.5)       # (1+Global_Mod) 乘子裁剪区间
TW_CMOD_CLIP = (0.5, 1.5)       # (1+Chain_Mod) 乘子裁剪区间

# ---- 撮合引擎 BacktestEngine ----
DEADZONE_TH = 0.05   # 调仓死区：已持仓 |Δ权重| < 该值跳过微调（建仓/清仓豁免）

# ---- A 股硬过滤层 ----
TH_AMOUNT = 1e7          # 分钟成交额门槛（元）
START_TIME = "09:45"     # 开仓时间窗起点
END_TIME = "14:50"       # 开仓时间窗终点

# ---- 历史兼容参数（旧二值化闸门阈值，新决策不再使用，仅保留构造兼容）----
TH_GLOBAL_MIN = 0.0
TH_ADR_MIN = 1.0
TH_MRS_MIN = 0.0
TH_INDUSTRY_MIN = 0.0
TH_MS_BULL = 0.0
TH_LOCK = 0.5
TH_CHASE = 0.7
TH_PURITY = 0.0
TH_MS_EXIT = -0.1
TH_SLIPPAGE = 0.03
WIN_HOLD_MAX = 240
TH_GRS_CIRCUIT = -1.5
