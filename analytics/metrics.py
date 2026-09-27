"""绩效指标：夏普比率、最大回撤、胜率、盈亏比与交易统计。

指标口径约定：
- 年化夏普：总权益按日末值重采样 → 日收益，
  年化 = (日收益均值 × 年交易日) / (日收益样本标准差 × sqrt(年交易日))，
  默认 244 个 A 股交易日；分钟级曲线中的 0 收益（无持仓时段）不参与日频口径。
- 最大回撤：对分钟级总权益曲线计算 (累计峰值 - 当前值) / 累计峰值 的最大值，
  返回非负比例（如 0.12 表示 -12%）。
- 胜率 / 盈亏比 / 有效交易笔数：从 TradeLog 按标的做 FIFO 加权平均成本配对，
  每笔平仓（filled SELL / t1_deferred_sell）实现一次已实现盈亏：
      realized_pnl = 卖出毛额 - 卖出费用 - 卖出份额 × 买入加权平均成本
  有效交易笔数 = 平仓笔数；胜率 = 盈利笔数 / 总笔数；
  盈亏比 = 平均盈利 / |平均亏损|（无亏损时视为 +inf）。

所有函数均为纯函数，输入为引擎产出的 DataFrame（TradeLog / EquityCurve），
便于单元测试与 Optuna 目标函数复用。
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 244

# 卖出成交的 reason（平仓记一笔已实现盈亏）
_SELL_REASONS = ("signal_sell", "t1_deferred_sell")


# ----------------------------------------------------------------------
# 净值曲线指标
# ----------------------------------------------------------------------


def _equity_series(equity_curve: pd.DataFrame) -> pd.Series:
    """提取升序、去重的总权益序列。"""
    if not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty:
        raise ValueError("equity_curve 必须是非空 DataFrame")
    if "total_equity" not in equity_curve.columns:
        raise ValueError("equity_curve 缺少 total_equity 列")
    s = equity_curve.set_index("ts")["total_equity"].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def daily_sharpe(equity_curve: pd.DataFrame, rf: float = 0.0,
                 periods_per_year: int = TRADING_DAYS) -> float:
    """年化夏普比率（日频重采样）。数据不足或收益零波动时返回 NaN。"""
    s = _equity_series(equity_curve)
    daily = s.resample("D").last().dropna()
    ret = daily.pct_change().dropna()
    if len(ret) < 2:
        return float("nan")
    std = ret.std(ddof=1)
    if not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    mean = ret.mean()
    return float((mean - rf / periods_per_year) / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.DataFrame) -> float:
    """最大回撤（分钟级曲线，返回非负比例）。"""
    s = _equity_series(equity_curve)
    if len(s) < 2:
        return 0.0
    peak = s.cummax()
    dd = (peak - s) / peak
    return float(dd.max()) if len(dd) else 0.0


def daily_returns(equity_curve: pd.DataFrame) -> pd.Series:
    """日频收益序列（日末重采样 pct_change）。"""
    s = _equity_series(equity_curve)
    return s.resample("D").last().dropna().pct_change().dropna()


def annual_return(equity_curve: pd.DataFrame,
                  periods_per_year: int = TRADING_DAYS) -> float:
    """年化收益率：日末值首末比按 244 个交易日年化。"""
    s = _equity_series(equity_curve)
    daily = s.resample("D").last().dropna()
    if len(daily) < 2 or daily.iloc[0] <= 0:
        return float("nan")
    total = float(daily.iloc[-1] / daily.iloc[0] - 1.0)
    if total <= -1.0:
        return -1.0
    n = len(daily)
    return float((1.0 + total) ** (periods_per_year / n) - 1.0)


def calmar_ratio(equity_curve: pd.DataFrame,
                 periods_per_year: int = TRADING_DAYS) -> float:
    """卡玛比率 = 年化收益率 / 最大回撤。零回撤且盈利时返回 +inf。"""
    ar = annual_return(equity_curve, periods_per_year)
    mdd = max_drawdown(equity_curve)
    if not np.isfinite(ar) or mdd <= 1e-12:
        return float("inf") if np.isfinite(ar) and ar > 0 else float("nan")
    return float(ar / mdd)


def sortino_ratio(equity_curve: pd.DataFrame, rf: float = 0.0,
                  periods_per_year: int = TRADING_DAYS) -> float:
    """Sortino 比率：以负收益样本标准差为分母的年化比率。"""
    ret = daily_returns(equity_curve)
    if len(ret) < 2:
        return float("nan")
    downside = ret[ret < 0]
    if len(downside) < 1:
        # 无亏损日：正收益视为无穷好，否则不可定义
        return float("inf") if ret.mean() > 0 else float("nan")
    dd = float(downside.std(ddof=1))
    if not np.isfinite(dd) or dd <= 1e-12:
        return float("nan")
    return float((ret.mean() - rf / periods_per_year) / dd
                 * np.sqrt(periods_per_year))


def return_skew_kurtosis(equity_curve: pd.DataFrame) -> Tuple[float, float]:
    """日度收益偏度 / 峰度（Fisher 峰度，正态分布为 0）。"""
    ret = daily_returns(equity_curve)
    if len(ret) < 3:
        return float("nan"), float("nan")
    return float(ret.skew()), float(ret.kurtosis())


# ----------------------------------------------------------------------
# 成交日志指标
# ----------------------------------------------------------------------


BATCH_COLUMNS = ["symbol", "sell_fill_id", "buy_fill_id", "buy_signal_id",
                 "entry_ts", "exit_ts", "matched_shares", "allocated_pnl",
                 "buy_decision_ts"]


def _local_time(value):
    """统一为无时区的上海本地时间；不把 UTC 时刻直接去掉时区。"""
    if value is None or pd.isna(value):
        return pd.NaT
    ts = pd.Timestamp(value)
    return ts.tz_convert("Asia/Shanghai").tz_localize(None) if ts.tzinfo else ts


def _prepare_trade_log(trade_log):
    """按显式事件顺序处理；旧日志用输入行序生成内部成交键，不伪造信号来源。"""
    cols = ["ts", "symbol", "side", "shares", "amount", "commission",
            "stamp_duty", "transfer_fee", "fill_id", "signal_id", "decision_ts"]
    if trade_log is None or trade_log.empty:
        return pd.DataFrame(columns=cols)
    frame = trade_log.copy(deep=True)
    frame["ts"] = frame["ts"].map(_local_time)
    if frame["ts"].isna().any():
        raise ValueError("成交时间缺失")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["_input_order"] = np.arange(len(frame))
    for col in ("shares", "amount", "commission", "stamp_duty", "transfer_fee"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if frame[col].isna().any() or not np.isfinite(frame[col]).all():
            raise ValueError(f"成交字段 {col} 必须为有限数值")
    if (frame.shares < 0).any() or (frame.shares % 1 != 0).any():
        raise ValueError("成交股数必须为非负整数")
    sort = ["ts", "_input_order"]
    if "event_seq" in frame:
        seq = pd.to_numeric(frame.event_seq, errors="raise")
        if not np.isfinite(seq).all() or (seq % 1 != 0).any() or seq.duplicated().any():
            raise ValueError("event_seq 必须是唯一整数")
        frame["event_seq"] = seq.astype("int64")
        sort = ["ts", "event_seq"]
    frame = frame.sort_values(sort, kind="stable").reset_index(drop=True)
    for col in ("fill_id", "signal_id"):
        if col not in frame:
            frame[col] = None
        # Pandas 的字符串推断会把 None 重新变成 NaN；明确保留 object 空 ID。
        frame[col] = pd.Series([None if pd.isna(v) or v == "" else v for v in frame[col]],
                               index=frame.index, dtype=object)
        if any(v is not None and not isinstance(v, str) for v in frame[col]):
            raise ValueError(f"{col} 必须为字符串")
    ids = frame.fill_id.dropna()
    if ids.duplicated().any():
        raise ValueError("重复 fill_id")
    used = set(ids)
    for i in frame.index[frame.shares.gt(0) & frame.fill_id.isna()]:
        key = f"legacy_fill_{i + 1:08d}"
        while key in used:
            key += "_legacy"
        frame.at[i, "fill_id"] = key
        used.add(key)
    if "decision_ts" not in frame:
        frame["decision_ts"] = pd.NaT
    frame["decision_ts"] = frame.decision_ts.map(_local_time)
    return frame


def _match_trade_batches(trade_log):
    """平均成本只计算一次盈亏；FIFO 队列仅分配股数来源，不使用批次原始买价。"""
    from collections import deque
    frame = _prepare_trade_log(trade_log)
    trades, batches = [], []
    remaining_shares = 0
    for sym, group in frame[frame.shares.gt(0)].groupby("symbol", sort=True):
        qty, cost = 0, 0.0
        lots = deque()
        for row in group.to_dict("records"):
            shares = int(row["shares"])
            if row["side"] in ("BUY", "ADD"):
                qty += shares
                cost += row["amount"] + row["commission"] + row["transfer_fee"]
                lots.append(dict(row, remaining=shares))
                continue
            if row["side"] != "SELL":
                raise ValueError(f"未知成交方向: {row['side']}")
            if shares > qty:
                raise ValueError(f"{sym} 超卖: {shares} > {qty}")
            avg = cost / qty
            proceeds = row["amount"] - row["commission"] - row["stamp_duty"] - row["transfer_fee"]
            pnl = float(proceeds - shares * avg)
            trades.append(dict(symbol=sym, entry_ts=lots[0]["ts"], exit_ts=row["ts"],
                               shares=float(shares), entry_price=float(avg),
                               proceeds=float(proceeds), pnl=pnl, sell_fill_id=row["fill_id"]))
            left, allocated = shares, 0.0
            while left:
                lot = lots[0]
                matched = min(left, lot["remaining"])
                # 最后一行承接浮点余差，保证同一笔 SELL 的批次金额严格守恒。
                part = pnl - allocated if matched == left else pnl * matched / shares
                batches.append(dict(symbol=sym, sell_fill_id=row["fill_id"],
                                    buy_fill_id=lot["fill_id"], buy_signal_id=lot["signal_id"],
                                    entry_ts=lot["ts"], exit_ts=row["ts"],
                                    matched_shares=matched, allocated_pnl=part,
                                    buy_decision_ts=lot["decision_ts"]))
                allocated += part
                left -= matched
                lot["remaining"] -= matched
                if not lot["remaining"]:
                    lots.popleft()
            qty -= shares
            cost = cost - shares * avg if qty else 0.0
        remaining_shares += qty
    return trades, pd.DataFrame(batches, columns=BATCH_COLUMNS), remaining_shares


def match_trade_batches(trade_log: pd.DataFrame) -> pd.DataFrame:
    """每行是一笔卖出所消耗的买入批次；allocated_pnl 是平均成本盈亏的股数分摊。"""
    return _match_trade_batches(trade_log)[1]


def closed_trades(trade_log: pd.DataFrame) -> List[Dict[str, object]]:
    """每笔 SELL 一行，保留平均成本及最早来源批次时间；与归因共用配对算法。"""
    return _match_trade_batches(trade_log)[0]


def holding_period(trade_log: pd.DataFrame, unit: str = "minutes") -> float:
    """平均持仓周期（默认分钟）。无平仓记录时返回 NaN。"""
    trades = closed_trades(trade_log)
    if not trades:
        return float("nan")
    scale = {"minutes": 1.0, "hours": 1 / 60.0, "days": 1 / 1440.0}
    durs = [(t["exit_ts"] - t["entry_ts"]).total_seconds()
            / 60.0 * scale.get(unit, 1.0) for t in trades]
    return float(np.mean(durs))


def trade_stats(trade_log: pd.DataFrame) -> Dict[str, float]:
    """从 TradeLog 计算交易统计（FIFO 加权平均成本配对）。

    :return: {n_trades, win_rate, profit_loss_ratio, total_pnl}
        - n_trades 为平仓笔数（有效交易）；无成交时 win_rate/pl_ratio 为 NaN
    """
    empty = {"n_trades": 0, "win_rate": float("nan"),
             "profit_loss_ratio": float("nan"), "total_pnl": 0.0}
    trades = closed_trades(trade_log)
    if not trades:
        return empty

    pnls = [t["pnl"] for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n
    pl_ratio = (float(np.mean(wins)) / abs(float(np.mean(losses)))
                if wins and losses
                else float("inf"))
    return {"n_trades": n, "win_rate": float(win_rate),
            "profit_loss_ratio": float(pl_ratio), "total_pnl": float(np.sum(pnls))}


# ----------------------------------------------------------------------
# 综合评估与约束
# ----------------------------------------------------------------------


def evaluate(equity_curve: pd.DataFrame, trade_log: pd.DataFrame) -> Dict[str, float]:
    """完整绩效评估：返回全部优化/约束所需的指标。"""
    st = trade_stats(trade_log)
    turnover, turnover_annual = _turnover_metrics(trade_log, equity_curve)
    return {
        "sharpe": daily_sharpe(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "n_trades": float(st["n_trades"]),
        "win_rate": float(st["win_rate"]),
        "profit_loss_ratio": float(st["profit_loss_ratio"]),
        "total_pnl": float(st["total_pnl"]),
        # 双边换手：成交额(买+卖) / 期间平均权益（倍数）；年化按 244 日折算。
        # 高频换手是 A 股实测最大亏损源之一，作为寻优软惩罚与硬约束的输入
        "turnover": turnover,
        "turnover_annual": turnover_annual,
    }


def _turnover_metrics(trade_log: pd.DataFrame,
                      equity_curve: pd.DataFrame) -> Tuple[float, float]:
    """双边换手率（倍/区间）与年化换手率。

    turnover = Σ成交额(买入+卖出) / mean(总权益)；turnover_annual = turnover × 244 / 交易日数。
    无成交时返回 (0.0, 0.0)；权益均值非法时返回 NaN。
    """
    if trade_log is None or len(trade_log) == 0:
        return 0.0, 0.0
    filled = trade_log[trade_log["shares"] > 0]
    if filled.empty or "amount" not in filled.columns:
        return 0.0, 0.0
    notional = float(filled["amount"].astype(float).sum())
    s = _equity_series(equity_curve)
    avg_eq = float(s.mean())
    if not np.isfinite(avg_eq) or avg_eq <= 1e-9:
        return float("nan"), float("nan")
    days = len(s.resample("D").last().dropna())
    turnover = notional / avg_eq
    annual = turnover * TRADING_DAYS / days if days else float("nan")
    return float(turnover), float(annual)


def constraint_violations(metrics: Dict[str, float],
                          max_drawdown: float = 0.15,
                          win_rate: float = 0.55,
                          pl_ratio: float = 1.5,
                          min_trades: int = 30,
                          max_turnover_annual: float = 1e9) -> List[float]:
    """硬约束违反量（>= 0，0 = 满足；NaN 视为最大违反 1e9）。

    返回顺序与 Optuna constraints_func 约定一致：
        [回撤, 胜率, 盈亏比, 交易笔数, 年化换手]
    """
    def _v(value: float, limit: float, direction: str) -> float:
        """从指标映射中读取有限数值；缺失、非数值或无穷值回退到调用方给定的默认值。"""
        v = float(value)
        if not np.isfinite(v):
            return 1e9
        if direction == "lt":
            return max(0.0, v - limit)      # 要求 v < limit
        return max(0.0, limit - v)          # 要求 v > limit

    return [
        _v(metrics.get("max_drawdown", float("nan")), max_drawdown, "lt"),
        _v(metrics.get("win_rate", float("nan")), win_rate, "gt"),
        _v(metrics.get("profit_loss_ratio", float("nan")), pl_ratio, "gt"),
        _v(metrics.get("n_trades", float("nan")), float(min_trades), "gt"),
        _v(metrics.get("turnover_annual", float("nan")),
           float(max_turnover_annual), "lt"),
    ]
