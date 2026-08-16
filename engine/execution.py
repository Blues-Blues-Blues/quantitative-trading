"""A 股交易成本与动态滑点模型 ExecutionCost。

成本明细：
- 佣金：双边收取，commission_rate（默认万分之二），单笔最低 min_commission（默认 5 元）
- 印花税：仅卖出收取，stamp_duty_rate（默认千分之 0.5）
- 过户费：双边收取，transfer_fee_rate（默认万分之一，即 0.00001）

动态滑点模型（基于当前 Bar 的成交量与 VWAP 偏差）：
    参与率 participation = 订单金额 / Bar 成交额（amount），clip 到 [0, 1]
    滑点 bps = fixed_slippage_bps + slippage_coef_bps × participation，封顶 slippage_cap_bps
    买入成交价 = open × (1 + 滑点)；卖出成交价 = open × (1 - 滑点)

说明：Bar 内 VWAP 与成交量的关系近似以「订单占 Bar 成交额比例」度量冲击成本，
参与率越高，滑点越大 —— 与实盘中大单吃穿盘口的行为一致。
"""

from typing import Tuple


class ExecutionCost:
    """交易成本与滑点模型（纯函数，可被 Optuna 寻优参数化）。"""

    def __init__(
        self,
        commission_rate: float = 0.0002,      # 佣金率（双边，万分之二）
        min_commission: float = 5.0,          # 单笔最低佣金（元）
        stamp_duty_rate: float = 0.0005,      # 印花税（仅卖出，千分之 0.5）
        transfer_fee_rate: float = 0.00001,   # 过户费（双边，万分之一）
        fixed_slippage_bps: float = 2.0,      # 固定滑点（基点）
        slippage_coef_bps: float = 50.0,      # 参与率敏感系数（基点 / 100% 参与率）
        slippage_cap_bps: float = 60.0,       # 滑点封顶（基点）
    ) -> None:
        if not 0.0 < commission_rate < 0.01:
            raise ValueError(f"commission_rate 必须在 (0, 0.01)，当前: {commission_rate}")
        if min_commission < 0:
            raise ValueError(f"min_commission 不能为负，当前: {min_commission}")
        if not 0.0 <= stamp_duty_rate < 0.05:
            raise ValueError(f"stamp_duty_rate 必须在 [0, 0.05)，当前: {stamp_duty_rate}")
        if not 0.0 <= transfer_fee_rate < 0.01:
            raise ValueError(f"transfer_fee_rate 必须在 [0, 0.01)，当前: {transfer_fee_rate}")
        if fixed_slippage_bps < 0 or slippage_coef_bps < 0:
            raise ValueError("滑点参数不能为负")
        if slippage_cap_bps < fixed_slippage_bps:
            raise ValueError(f"slippage_cap_bps 不能小于固定滑点，当前: "
                             f"{slippage_cap_bps} < {fixed_slippage_bps}")

        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.fixed_slippage_bps = fixed_slippage_bps
        self.slippage_coef_bps = slippage_coef_bps
        self.slippage_cap_bps = slippage_cap_bps

    # ------------------------------------------------------------------
    # 滑点模型
    # ------------------------------------------------------------------

    def participation(self, order_amount: float, bar_amount: float) -> float:
        """订单占该 Bar 成交额的比例（clip [0,1]；无成交量时视为 0）。"""
        if bar_amount <= 0:
            return 0.0
        return min(1.0, max(0.0, order_amount / bar_amount))

    def slippage_bps(self, order_amount: float, bar_amount: float) -> float:
        """动态滑点（基点）：固定项 + 参与率冲击项，封顶。"""
        part = self.participation(order_amount, bar_amount)
        bps = self.fixed_slippage_bps + self.slippage_coef_bps * part
        return min(bps, self.slippage_cap_bps)

    def slippage(self, order_amount: float, bar_amount: float) -> float:
        """滑点比例（小数）。"""
        return self.slippage_bps(order_amount, bar_amount) / 1e4

    def buy_price(self, open_price: float, order_amount: float,
                  bar_amount: float) -> float:
        """买入成交价 = open × (1 + 滑点)。"""
        assert open_price > 0
        return open_price * (1.0 + self.slippage(order_amount, bar_amount))

    def sell_price(self, open_price: float, order_amount: float,
                   bar_amount: float) -> float:
        """卖出成交价 = open × (1 - 滑点)。"""
        assert open_price > 0
        return open_price * (1.0 - self.slippage(order_amount, bar_amount))

    # ------------------------------------------------------------------
    # 费用
    # ------------------------------------------------------------------

    def buy_fees(self, amount: float) -> Tuple[float, float]:
        """买入费用：(佣金, 过户费)。"""
        commission = max(amount * self.commission_rate, self.min_commission)
        transfer = amount * self.transfer_fee_rate
        return commission, transfer

    def sell_fees(self, amount: float) -> Tuple[float, float, float]:
        """卖出费用：(佣金, 印花税, 过户费)。"""
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp = amount * self.stamp_duty_rate
        transfer = amount * self.transfer_fee_rate
        return commission, stamp, transfer
