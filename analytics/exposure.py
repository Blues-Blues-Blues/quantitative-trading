"""收盘策略分数加权暴露；保留现金影响、因子符号及缺失覆盖率，不估计 beta。"""

import numpy as np
import pandas as pd

from analytics.metrics import _local_time

FACTORS = ["global_mod", "chain_mod", "agent_ms"]
POSITION_COLUMNS = ["ts", "symbol", "shares", "mark_price", "market_value", "total_equity"]
CONTRIBUTION_COLUMNS = ["ts", "symbol", "factor", "shares", "market_value",
                        "position_weight", "factor_value", "contribution"]
EXPOSURE_COLUMNS = ["ts", "factor", "exposure", "observed_exposure", "coverage",
                   "status", "invested_weight"]


class ExposureAnalyzer:
    """纯分析接口：所有关联均为同一收盘时刻，不对缺失因子做前向/后向填充。"""

    @staticmethod
    def compute(positions, factor_values, equity_curve):
        positions = positions.copy(deep=True)
        values = factor_values.copy(deep=True).reindex(columns=["ts", "symbol"] + FACTORS)
        equity = equity_curve[["ts", "total_equity"]].copy()
        for table in (positions, values, equity):
            table["ts"] = pd.to_datetime(table.ts.map(_local_time))
            if table.ts.isna().any():
                raise ValueError("暴露输入 ts 缺失")
            if "symbol" in table:
                table["symbol"] = table.symbol.astype(str)
            keys = ["ts", "symbol"] if "symbol" in table else ["ts"]
            if table.duplicated(keys).any():
                raise ValueError(f"暴露输入包含重复键 {keys}")
        equity["total_equity"] = pd.to_numeric(equity.total_equity, errors="raise")
        if equity.total_equity.isna().any() or not np.isfinite(equity.total_equity).all() or equity.total_equity.le(0).any():
            raise ValueError("总权益必须为有限正数")
        if not positions.ts.isin(equity.ts).all():
            raise ValueError("持仓时点不在净值时间轴中")
        for column in ("shares", "market_value", "total_equity"):
            positions[column] = pd.to_numeric(positions[column], errors="raise")
            if positions[column].isna().any() or not np.isfinite(positions[column]).all() or positions[column].lt(0).any():
                raise ValueError(f"无效持仓字段 {column}")
        positions = positions[positions.shares.gt(0)]
        merged = positions.merge(equity, on="ts", suffixes=("_position", ""), validate="many_to_one")
        if not np.allclose(merged.total_equity_position, merged.total_equity, rtol=0, atol=1e-6):
            raise ValueError("持仓与净值表总权益矛盾")
        merged = merged.merge(values, on=["ts", "symbol"], how="left", validate="one_to_one")
        merged["position_weight"] = merged.market_value / merged.total_equity
        long = merged.melt(id_vars=["ts", "symbol", "shares", "market_value", "position_weight"],
                           value_vars=FACTORS, var_name="factor", value_name="factor_value")
        long["factor_value"] = pd.to_numeric(long.factor_value, errors="coerce").astype(float)
        long.loc[~np.isfinite(long.factor_value), "factor_value"] = np.nan
        long["contribution"] = long.position_weight * long.factor_value
        contributions = long.reindex(columns=CONTRIBUTION_COLUMNS)
        rows = []
        by_ts = {ts: group for ts, group in contributions.groupby("ts", sort=False)}
        for point in equity.sort_values("ts").itertuples(index=False):
            group = by_ts.get(point.ts)
            for factor in FACTORS:
                g = group[group.factor.eq(factor)] if group is not None else contributions.iloc[:0]
                if g.empty:
                    observed, exposure, coverage, invested, status = 0.0, 0.0, 1.0, 0.0, "flat"
                else:
                    market_value = float(g.market_value.sum())
                    coverage = float(g.loc[g.factor_value.notna(), "market_value"].sum() / market_value) if market_value else 1.0
                    observed = float(g.contribution.sum())
                    invested = market_value / point.total_equity
                    status = "complete" if coverage >= 1 - 1e-12 else "partial"
                    exposure = observed if status == "complete" else np.nan
                rows.append(dict(ts=point.ts, factor=factor, exposure=exposure,
                                 observed_exposure=observed, coverage=coverage,
                                 status=status, invested_weight=invested))
        return contributions, pd.DataFrame(rows, columns=EXPOSURE_COLUMNS)
