"""动态仓位管理 PositionSizer：Position_Size 公式与仓位/杠杆风控。

核心公式（信号层给出的目标仓位）：
    Position_Size = Base_Position × MRS_Coefficient × (1 + Global_Mod) × Chain_Mod_Scale

    MRS_Coefficient : 大盘共振系数，由 MRS 线性映射并 clip 到 [mrs_coef_min, mrs_coef_max]
                      coef = 1 + mrs_sensitivity × MRS
    Chain_Mod_Scale : 链式调节系数 = max(chain_scale_floor, 1 + Chain_Mod)
    Base_Position   : 单标的基准仓位（占权益比例）

硬性风控（撮合引擎在成交前调用 Account 层校验）：
- 单股最大仓位上限：单标的市值 ≤ max_single_position × 总权益
- 总账户杠杆上限：全部市值 ≤ max_leverage × 总权益

熔断机制（大盘/外盘）由信号层平仓闸门 ⑥ 负责，本模块只负责"买多少"的仓位约束。
"""

from typing import Dict, Optional, Tuple


class PositionSizer:
    """动态仓位计算器（纯函数，全部参数可被 Optuna 寻优）。"""

    def __init__(
        self,
        base_position: float = 0.20,      # 基准仓位（占权益比例）
        max_single_position: float = 0.30,  # 单股最大仓位（占权益比例，硬上限）
        max_leverage: float = 1.0,        # 总杠杆上限（1.0 = 无杠杆）
        mrs_coef_min: float = 0.5,        # MRS 系数下界
        mrs_coef_max: float = 1.5,        # MRS 系数上界
        mrs_sensitivity: float = 10.0,    # MRS → 系数 的线性灵敏度
        chain_scale_floor: float = 0.5,   # 链式调节系数下界
    ) -> None:
        if not 0.0 < base_position <= 1.0:
            raise ValueError(f"base_position 必须在 (0, 1] 区间，当前: {base_position}")
        if not 0.0 < max_single_position <= 1.0:
            raise ValueError(f"max_single_position 必须在 (0, 1] 区间，当前: {max_single_position}")
        if max_single_position < base_position:
            raise ValueError("max_single_position 不能小于 base_position（上限应不小于基准）")
        if max_leverage < 1.0:
            raise ValueError(f"max_leverage 必须 >= 1.0，当前: {max_leverage}")
        if not 0.0 < mrs_coef_min <= mrs_coef_max:
            raise ValueError(f"MRS 系数区间非法: [{mrs_coef_min}, {mrs_coef_max}]")
        if chain_scale_floor <= 0:
            raise ValueError(f"chain_scale_floor 必须为正，当前: {chain_scale_floor}")

        self.base_position = base_position
        self.max_single_position = max_single_position
        self.max_leverage = max_leverage
        self.mrs_coef_min = mrs_coef_min
        self.mrs_coef_max = mrs_coef_max
        self.mrs_sensitivity = mrs_sensitivity
        self.chain_scale_floor = chain_scale_floor

    # ------------------------------------------------------------------
    # 公式分量
    # ------------------------------------------------------------------

    def mrs_coefficient(self, mrs: Optional[float]) -> float:
        """MRS 系数 = clip(1 + sensitivity × MRS, min, max)；MRS 缺失视为中性 1.0。"""
        m = 0.0 if mrs is None else float(mrs)
        return min(self.mrs_coef_max, max(self.mrs_coef_min, 1.0 + self.mrs_sensitivity * m))

    def chain_scale(self, chain_mod: Optional[float]) -> float:
        """链式调节 = max(floor, 1 + Chain_Mod)；缺失视为中性 1.0。"""
        c = 0.0 if chain_mod is None else float(chain_mod)
        return max(self.chain_scale_floor, 1.0 + c)

    # ------------------------------------------------------------------
    # 目标仓位
    # ------------------------------------------------------------------

    def target_ratio(self, metrics: Dict[str, object]) -> float:
        """目标仓位比例（相对总权益）。

        Position_Size = Base_Position × MRS_Coefficient × (1 + Global_Mod) × Chain_Mod_Scale
        结果 clip 到 [0, max_single_position]（单股上限兜底）。

        :param metrics: 信号附带的指标快照（Signal.metrics），需含
                        mrs / global_mod / chain_mod（缺失视为中性 0）
        """
        mrs = metrics.get("mrs")
        gmod = metrics.get("global_mod")
        chain = metrics.get("chain_mod")
        g = 0.0 if gmod is None else float(gmod)

        ratio = self.base_position * self.mrs_coefficient(mrs) * (1.0 + g) * self.chain_scale(chain)
        # 单股上限兜底：目标比例不得超过 max_single_position
        return min(self.max_single_position, max(0.0, ratio))

    def target_value(self, equity: float, metrics: Dict[str, object]) -> float:
        """目标市值（元）= 目标比例 × 总权益。"""
        assert equity > 0, f"总权益必须为正，当前: {equity}"
        return self.target_ratio(metrics) * equity

    # ------------------------------------------------------------------
    # 风控校验（返回 (是否通过, 拒绝原因)；None 表示通过）
    # ------------------------------------------------------------------

    def check_single_position(self, equity: float, current_value: float,
                              order_value: float) -> Tuple[bool, Optional[str]]:
        """单股上限：现有该股市值 + 本次加仓 ≤ max_single_position × 权益。"""
        limit = self.max_single_position * equity
        ok = current_value + order_value <= limit + 1e-6
        return ok, None if ok else "max_single_position"

    def check_leverage(self, equity: float, position_value: float,
                       order_value: float) -> Tuple[bool, Optional[str]]:
        """总杠杆上限：现有市值 + 本次下单 ≤ max_leverage × 权益。"""
        limit = self.max_leverage * equity
        ok = position_value + order_value <= limit + 1e-6
        return ok, None if ok else "leverage_cap"
