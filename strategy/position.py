"""仓位管理：持仓限制（最大持股数、总仓位上限）与目标权重调整。

硬性约束：
- 最大持股数不超过 max_holdings（默认 4 只）
- 所有持仓总权重不得超过 max_total_weight（默认 0.95，即至少保留 5% 现金）

该模块只负责"约束"，不负责选股与信号判断，选股结果由策略层提供。
"""

from typing import Dict, Sequence


class PositionManager:
    """仓位管理器：对策略给出的目标权重强制执行持仓约束。

    用法：
        pm = PositionManager(max_holdings=4, max_total_weight=0.95)
        targets = {"600000": 0.3, "000001": 0.3, "000002": 0.3, "600519": 0.2, "601398": 0.2}
        result = pm.adjust(targets)   # 最多保留 4 只，总权重 <= 0.95
        # result 的权重总和不足 1 的部分即为保留的现金比例
    """

    def __init__(self, max_holdings: int = 4, max_total_weight: float = 0.95):
        if max_holdings <= 0:
            raise ValueError(f"max_holdings 必须为正整数，当前值: {max_holdings}")
        if not 0.0 < max_total_weight <= 1.0:
            raise ValueError(f"max_total_weight 必须在 (0, 1] 区间，当前值: {max_total_weight}")
        self.max_holdings = max_holdings
        self.max_total_weight = max_total_weight

    @property
    def cash_ratio(self) -> float:
        """强制保留的最低现金比例。"""
        return 1.0 - self.max_total_weight

    def adjust(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """按约束调整目标权重，返回实际可用权重。

        规则：
        1. 剔除权重 <= 0 的标的（视为不持仓）
        2. 按权重降序仅保留前 max_holdings 只（候选股过多时只买最看好的）
        3. 若总权重超过 max_total_weight，按比例等比压缩，保证现金 >= 5%

        :param target_weights: {symbol: 目标权重}，权重为 0~1 的小数
        :return: {symbol: 调整后权重}，未满仓部分即现金
        """
        weights = {k: v for k, v in target_weights.items() if v > 0}
        if not weights:
            return {}

        # 1) 按权重降序，最多保留 max_holdings 只
        ranked = dict(sorted(weights.items(), key=lambda kv: kv[1], reverse=True))
        selected = dict(list(ranked.items())[: self.max_holdings])

        # 2) 总权重超限时按比例压缩
        total = sum(selected.values())
        if total > self.max_total_weight:
            scale = self.max_total_weight / total
            selected = {k: v * scale for k, v in selected.items()}

        return selected

    def can_add(self, current_symbols: Sequence[str]) -> bool:
        """当前持仓数是否还能继续买入新股票（未达到最大持股数）。"""
        return len(current_symbols) < self.max_holdings
