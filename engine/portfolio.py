"""账户状态机 Account：现金 / 总权益 / 融资 / 持仓（含 A 股 T+1 可卖份额）管理。

T+1 规则：当日买入的份额当日不可卖出。
持仓以两个份额字段跟踪：
    shares            总持仓份额（= 已解冻可卖 + 当日新买未解冻）
    sellable_shares   当前可卖出份额（买入当日不增加，次日开市由 roll_to_date 解冻）

杠杆预留：margin 为融资余额（默认 0），max_leverage 限制总市值 / 总权益 ≤ 杠杆上限。
现金扣减与入账均在 Account 内完成；成本（佣金/印花税/过户费）由调用方（撮合引擎）
计算后以净额传入，Account 不感知费用明细，职责单一。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class Position:
    """单个标的的账户持仓。"""

    symbol: str
    shares: int = 0                # 总持仓份额
    sellable_shares: int = 0       # 当前可卖出份额（T+1 解冻后）
    cost_basis: float = 0.0        # 加权平均成本价
    last_price: float = 0.0        # 最近一次标记价格（收盘 mark）
    last_buy_date: Optional[pd.Timestamp] = None  # 最近一次买入的交易日（T+1 判定）

    @property
    def market_value(self) -> float:
        """按最近标记价计算的持仓市值。"""
        return self.shares * self.last_price

    @property
    def locked_shares(self) -> int:
        """当日新买、尚未解冻（不可卖）的份额。"""
        return self.shares - self.sellable_shares

    def __repr__(self) -> str:  # 便于日志/复盘
        return (f"Position({self.symbol}, shares={self.shares}, "
                f"sellable={self.sellable_shares}, cost={self.cost_basis:.4f}, "
                f"last={self.last_price:.4f})")


class Account:
    """账户状态机：Cash / Total_Equity / Margin / Positions。

    :param initial_cash:        初始现金
    :param max_leverage:        总市值 / 总权益上限（1.0 = 无杠杆，不允许融资）
    :param max_single_position: 单股最大市值占权益比例上限（硬约束，撮合时校验）
    """

    def __init__(self, initial_cash: float, max_leverage: float = 1.0,
                 max_single_position: float = 0.30) -> None:
        if initial_cash <= 0:
            raise ValueError(f"initial_cash 必须为正，当前: {initial_cash}")
        if max_leverage < 1.0:
            raise ValueError(f"max_leverage 必须 >= 1.0（1.0 表示无杠杆），当前: {max_leverage}")
        if not 0.0 < max_single_position <= 1.0:
            raise ValueError(f"max_single_position 必须在 (0, 1] 区间，当前: {max_single_position}")

        self.initial_cash = float(initial_cash)
        self.max_leverage = float(max_leverage)
        self.max_single_position = float(max_single_position)

        self.cash: float = float(initial_cash)
        self.margin: float = 0.0            # 融资余额（当前默认 0，预留杠杆通道）
        self.positions: Dict[str, Position] = {}

    # ------------------------------------------------------------------
    # 净值
    # ------------------------------------------------------------------

    @property
    def position_value(self) -> float:
        """全部持仓市值之和。"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_equity(self) -> float:
        """总权益 = 现金 + 持仓市值 - 融资余额。"""
        return self.cash + self.position_value - self.margin

    @property
    def realized_pnl(self) -> float:
        """已实现盈亏（近似：初始现金变化即已实现部分，未包含浮动盈亏）。"""
        return self.cash - self.initial_cash + self.margin

    def unrealized_pnl(self) -> float:
        """浮动盈亏 = 持仓市值 - 持仓成本。"""
        return sum(p.market_value - p.cost_basis * p.shares
                   for p in self.positions.values())

    # ------------------------------------------------------------------
    # T+1 解冻
    # ------------------------------------------------------------------

    def roll_to_date(self, date) -> None:
        """新交易日开市调用：将上一交易日买入的份额全部解冻为可卖。

        :param date: 当前 Bar 的交易日（datetime / Timestamp 均可）
        """
        date = pd.Timestamp(date).normalize()
        for pos in self.positions.values():
            if pos.last_buy_date is not None and date > pos.last_buy_date:
                pos.sellable_shares = pos.shares

    # ------------------------------------------------------------------
    # 交易执行（撮合引擎调用）
    # ------------------------------------------------------------------

    def buy(self, symbol: str, ts: pd.Timestamp, price: float,
            shares: int, cost_total: float) -> None:
        """买入成交：扣现金、更新加权成本与持仓份额（当日买入不可卖）。

        :param shares:      成交份额（A 股买入必须 100 股整数倍，断言校验）
        :param cost_total:  成交金额 + 佣金 + 过户费等全部支出（净扣款）
        """
        assert shares > 0, f"买入份额必须为正，当前: {shares}"
        assert shares % 100 == 0, f"A 股买入必须为 100 股整数倍，当前: {shares}"
        assert price > 0, f"成交价必须为正，当前: {price}"
        assert cost_total >= 0, f"总成本不能为负，当前: {cost_total}"

        new_cash = self.cash - cost_total
        assert new_cash >= -1e-6, f"买入后现金为负: {new_cash:.2f} < 0"
        self.cash = new_cash

        pos = self.positions.get(symbol)
        if pos is None:
            pos = Position(symbol=symbol)
            self.positions[symbol] = pos

        total = pos.shares + shares
        pos.cost_basis = (pos.cost_basis * pos.shares + price * shares) / total
        pos.shares = total
        pos.last_price = float(price)
        pos.last_buy_date = pd.Timestamp(ts).normalize()
        # 注意：sellable_shares 不增加 —— 当日买入份额当日不可卖（T+1）

    def sell(self, symbol: str, ts: pd.Timestamp, price: float,
             shares: int, proceeds: float) -> None:
        """卖出成交：从可卖份额扣减、入账现金（净额 = 成交金额 - 各项费用）。

        :param shares:    卖出份额（不得超过 sellable_shares，断言校验）
        :param proceeds:  卖出净入账（含佣金/印花税/过户费扣除后的金额）
        """
        pos = self.positions.get(symbol)
        assert pos is not None, f"卖出时无持仓: {symbol}"
        assert 0 < shares <= pos.sellable_shares, (
            f"卖出份额 {shares} 超过可卖份额 {pos.sellable_shares}（T+1 约束）")
        assert price > 0, f"成交价必须为正，当前: {price}"

        self.cash += proceeds
        pos.shares -= shares
        pos.sellable_shares -= shares
        pos.last_price = float(price)
        if pos.shares <= 0:
            del self.positions[symbol]

    def mark_to_market(self, price_map: Dict[str, float]) -> None:
        """收盘/Bar 结束按最新价标记全部持仓（用于净值曲线）。"""
        for sym, price in price_map.items():
            pos = self.positions.get(sym)
            if pos is not None:
                pos.last_price = float(price)

    # ------------------------------------------------------------------
    # 约束校验（撮合前调用，返回 (是否通过, 拒绝原因)）
    # ------------------------------------------------------------------

    def check_leverage(self, additional_value: float) -> tuple:
        """总杠杆上限：现有市值 + 新增市值 ≤ max_leverage × 总权益。"""
        limit = self.max_leverage * self.total_equity
        ok = self.position_value + additional_value <= limit + 1e-6
        return ok, None if ok else "leverage_cap"

    def check_single_position(self, symbol: str, target_value: float) -> tuple:
        """单股上限：该标的持仓市值 ≤ max_single_position × 总权益。"""
        limit = self.max_single_position * self.total_equity
        current = self.positions.get(symbol).market_value if symbol in self.positions else 0.0
        ok = current + target_value <= limit + 1e-6
        return ok, None if ok else "max_single_position"
