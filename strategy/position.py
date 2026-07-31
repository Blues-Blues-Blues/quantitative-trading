"""仓位管理：持仓限制（最大持股数、总仓位上限）与目标权重调整。

硬性约束：
- 最大持股数不超过 max_holdings（默认 4 只）
- 所有持仓总权重不得超过 max_total_weight（默认 0.95，即至少保留 5% 现金）

目标仓位计算（compute_target_weights）：基础权重映射 + 波动率/MDM/震荡市修正 + 总仓封顶。

该模块只负责"约束与权重计算"，不负责选股与信号判断，选股结果由策略层提供。
"""

from typing import Dict, Sequence

# 目标权重映射：TQ+VC 总分 -> 基础权重（<=2 分为 0）
_BASE_WEIGHTS = {5: 0.25, 4: 0.18, 3: 0.10}

# MDM 修正阈值：权重 >= 18% 时降至 10%，否则降至 0%
_MDM_HIGH_FLOOR = 0.18
_MDM_DECAY_WEIGHT = 0.10

# 波动率修正系数
_VOLATILITY_FACTOR = 0.6


def weight_from_score(score: int) -> float:
    """基础权重映射：TQ+VC 总分（0~5）-> 基础权重。

    5分 -> 25%，4分 -> 18%，3分 -> 10%，<=2分 -> 0%。
    """
    return _BASE_WEIGHTS.get(int(score), 0.0)


def compute_target_weights(
    stock_factors: Dict[str, dict],
    max_total_weight: float = 0.95,
) -> Dict[str, float]:
    """目标仓位计算：按顺序执行波动率/MDM/震荡市修正，最后总仓封顶。

    :param stock_factors: {symbol: {"score": tq+vc 总分,
                                    "atr_ratio": ATR/Close,
                                    "thresh_atr": ATR 动态阈值,
                                    "mdm": 动量衰减是否生效（已含持仓>=5日门控）,
                                    "is_shock": 是否震荡市}}，
        其中 is_shock 在新指标体系下由 is_trend 反推：is_shock = not is_trend
        （is_trend 为 False 即视为震荡，不持仓）
    :param max_total_weight: 总权重上限（默认 0.95）
    :return: {symbol: 目标权重}，仅含权重 > 0 的标的；未满仓部分即现金
    """
    weights: Dict[str, float] = {}
    for symbol, f in stock_factors.items():
        w = weight_from_score(f["score"])

        # 1. 波动率修正：高波动（ATR/Close > 阈值）权重打 6 折
        if float(f["atr_ratio"]) > float(f["thresh_atr"]):
            w *= _VOLATILITY_FACTOR

        # 2. MDM 修正：动量衰减（且已持仓 >= 5 日）大幅降权
        if f.get("mdm"):
            w = _MDM_DECAY_WEIGHT if w >= _MDM_HIGH_FLOOR else 0.0

        # 3. 震荡市修正：震荡期不持仓
        if f.get("is_shock"):
            w = 0.0

        if w > 0:
            weights[symbol] = w

    # 4. 总仓封顶：所有个股权重和 > 上限时按比例缩放
    total = sum(weights.values())
    if total > max_total_weight:
        scale = max_total_weight / total
        weights = {s: w * scale for s, w in weights.items()}

    return weights


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
