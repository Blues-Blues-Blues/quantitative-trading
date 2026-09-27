"""只读的日频复盘数据。上海本地时间、元/股，百分比为比例值。

账户成本不含费；卖出盈亏复用 metrics 的含费平均成本口径。
不完整行情产生结构化警告；不一致的账户、快照或成交抛 ValueError。
"""
import math

import numpy as np
import pandas as pd

from analytics import metrics


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def _time_frame(frame, name, required):
    if not isinstance(frame, pd.DataFrame) or not set(required).issubset(frame.columns):
        raise ValueError(f"{name} 缺少字段 {required}")
    frame = frame.copy(deep=True).reset_index(drop=True)
    frame["ts"] = pd.to_datetime(frame.ts.map(metrics._local_time))
    if frame.ts.isna().any():
        raise ValueError(f"{name} 时间缺失")
    frame["date"] = frame.ts.dt.strftime("%Y-%m-%d")
    return frame.sort_values("ts", kind="stable")


def _numbers(frame, columns, name, positive=False):
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col]).all() or (positive and frame[col].le(0).any()):
            raise ValueError(f"{name}.{col} 必须是有限{'正' if positive else ''}数")


def _same(left, right, message):
    if not np.allclose(left, right, atol=1e-6, rtol=1e-10):
        raise ValueError(message)


def build_visualization_data(*, kline, trade_log, equity_curve, positions_daily,
                             symbols, initial_cash, metadata):
    """生成独立 JSON schema；kline 支持 DatetimeIndex 或 ts 列。

    positions_daily 必须密集且与每日最后账户 Bar 同时；初始空仓。
    输入不被修改，行情只裁剪本次账户范围，缺行情不会补成价格或市值零。
    """
    if initial_cash is None or not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash 必须显式提供有限正数")
    symbols = list(symbols)
    if not symbols or any(not isinstance(s, str) or not s.strip() for s in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError("symbols 必须是非空、无重复的代码字符串列表")
    eq = _time_frame(equity_curve, "equity_curve", ["ts", "total_equity", "position_value"])
    if eq.empty or eq.ts.duplicated().any():
        raise ValueError("账户曲线为空或账户时间重复")
    _numbers(eq, ["total_equity"], "账户权益", positive=True)
    _numbers(eq, ["position_value"], "账户市值")
    t0, t1 = eq.ts.iloc[0], eq.ts.iloc[-1]
    eq["worst_drawdown"] = eq.total_equity / eq.total_equity.cummax() - 1
    end = eq.groupby("date", sort=True).tail(1).set_index("date")
    days = end.index.tolist()
    pos = _time_frame(positions_daily, "positions_daily",
                      ["ts", "symbol", "shares", "mark_price", "market_value", "total_equity"])
    expected = {(d, s) for d in days for s in symbols}
    if pos.duplicated(["date", "symbol"]).any() or set(zip(pos.date, pos.symbol)) != expected:
        raise ValueError("日末快照必须完整且唯一：每日 × 每只报告股票；不支持未知期初持仓")
    _numbers(pos, ["shares", "market_value", "total_equity"], "持仓")
    if pos.shares.lt(0).any() or (pos.shares % 1 != 0).any():
        raise ValueError("持仓股数必须为非负整数")
    for p in pos.itertuples():
        e = end.loc[p.date]
        if p.ts != e.ts:
            raise ValueError(f"{p.symbol} {p.date} 日末快照时间与账户不一致")
        _same(p.total_equity, e.total_equity, f"{p.symbol} {p.date} 快照权益不一致")
        if p.shares:
            if pd.isna(p.mark_price) or not np.isfinite(p.mark_price) or p.mark_price <= 0:
                raise ValueError(f"{p.symbol} {p.date} 有仓标记价格非法")
            _same(p.market_value, p.shares * p.mark_price, f"{p.symbol} {p.date} 持仓市值不一致")
        else:
            _same(p.market_value, 0, f"{p.symbol} {p.date} 空仓市值非零")
    _same(pos.groupby("date").market_value.sum().reindex(days), end.position_value,
          "各股快照市值合计与账户 position_value 不一致")

    # 在完整日志上规范化一次，拒单参与旧 ID 的生成；复用同一算法，不能另编 ID。
    prepared = metrics._prepare_trade_log(trade_log)
    actual = prepared[prepared.shares.gt(0)].copy()
    if "event_seq" not in actual:
        actual["event_seq"] = np.arange(len(actual))
    if not actual.empty:
        _numbers(actual, ["price"], "实际成交价格", positive=True)
        if not actual.side.isin(["BUY", "ADD", "SELL"]).all():
            raise ValueError("未知实际成交方向")
        if not actual.symbol.isin(symbols).all() or not actual.ts.isin(eq.ts).all():
            raise ValueError("实际成交证券或时间不在账户回测集合中")
        _same(actual.amount, actual.price * actual.shares, "成交 amount 与 price × shares 不一致")
    closed = metrics.closed_trades(prepared)
    sell_ids = actual.loc[actual.side.eq("SELL"), "fill_id"].tolist()
    pnl_ids = [r["sell_fill_id"] for r in closed]
    if len(set(pnl_ids)) != len(pnl_ids) or set(pnl_ids) != set(sell_ids):
        raise ValueError("sell_fill_id 盈亏关联必须一一对应，无缺失、重复或额外记录")
    pnl = {r["sell_fill_id"]: r["pnl"] for r in closed}
    stats = metrics.evaluate(eq, prepared)
    _same(sum(pnl.values()), stats["total_pnl"], "已实现盈亏与 evaluate 不一致")
    state = {s: (0, None) for s in symbols}
    fills, daily_cost = [], {}
    events = iter(actual.to_dict("records"))
    row = next(events, None)
    for date, e in end.iterrows():
        while row is not None and row["ts"] <= e.ts:
            s, qty, price = row["symbol"], int(row["shares"]), row["price"]
            before, cost = state[s]
            if row["side"] in ("BUY", "ADD"):
                after = before + qty
                cost = ((cost or 0) * before + price * qty) / after
                action = "买入" if before == 0 else "加仓"
            else:
                if qty > before:
                    raise ValueError(f"{s} 超卖或未知期初持仓")
                after = before - qty
                action = "减仓" if after else "清仓"
                cost = cost if after else None
            state[s] = (after, cost)
            fills.append(dict(fill_id=row["fill_id"], event_seq=row["event_seq"], symbol=s,
                              ts=row["ts"], date=row["ts"].strftime("%Y-%m-%d"), side=row["side"],
                              action_type=action, price=price, shares=qty, amount=row["amount"],
                              fees=sum(row[c] for c in ("commission", "stamp_duty", "transfer_fee")),
                              realized_pnl=pnl.get(row["fill_id"]), shares_before=before,
                              shares_after=after, cost_basis_after=cost))
            row = next(events, None)
        for p in pos[pos.date.eq(date)].itertuples():
            if p.shares != state[p.symbol][0]:
                raise ValueError(f"{p.symbol} {date} 成交重建股数与快照不一致（不支持未知期初持仓）")
            daily_cost[(date, p.symbol)] = state[p.symbol][1]

    warnings = []
    def warn(code, message, symbol=None):
        warnings.append(dict(code=code, message=message, **({"symbol": symbol} if symbol else {})))

    if kline.empty:
        bars = pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    else:
        bars = kline.copy(deep=True)
        if "ts" not in bars:
            bars["ts"] = bars.index
    bars = _time_frame(bars, "kline", ["ts", "symbol"])
    bars = bars[bars.symbol.isin(symbols) & bars.ts.between(t0, t1)].copy()
    if bars.duplicated(["symbol", "ts"]).any():
        raise ValueError("行情存在重复 (symbol, timestamp)")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in bars:
            bars[col] = np.nan
        bars[col] = pd.to_numeric(bars[col], errors="raise")
        valid = bars[col].dropna()
        if not np.isfinite(valid).all() or (valid.le(0).any() if col != "volume" else valid.lt(0).any()):
            raise ValueError(f"行情 {col} 存在非正价格、负成交量或非有限值")
    if (bars.high.lt(bars.low) | bars.high.lt(bars.open) | bars.high.lt(bars.close)
            | bars.low.gt(bars.open) | bars.low.gt(bars.close)).any():
        raise ValueError("行情 OHLC 内部矛盾")
    stocks, bases, series, excluded = [], {}, {}, []
    for s in symbols:
        b = bars[bars.symbol.eq(s)]
        if b.empty or b.date.nunique() < len(days):
            warn("missing_market", "缺少部分或全部行情；缺口不填充，不推断停牌。", s)
        if b[["open", "high", "low", "close", "volume"]].isna().any().any():
            warn("incomplete_market", "部分行情字段缺失，按可用 Bar 聚合。", s)
        daily = b.groupby("date").agg(open=("open", "first"), high=("high", "max"),
                  low=("low", "min"), close=("close", "last"),
                  volume=("volume", lambda v: v.sum(min_count=1)),
                  first_bar_ts=("ts", "first"), last_bar_ts=("ts", "last"))
        for window in (5, 20, 60):
            daily[f"ma{window}"] = daily.close.dropna().rolling(window, min_periods=window).mean()
        daily = daily.reindex(days)
        p = pos[pos.symbol.eq(s)].set_index("date").reindex(days)
        for col in ("shares", "mark_price", "market_value"):
            daily[col] = p[col]
        daily["position_weight"] = p.market_value / p.total_equity
        daily["cost_basis"] = [daily_cost[(d, s)] for d in days]
        base = b.loc[b.ts.eq(t0), "open"]
        if base.empty or pd.isna(base.iloc[0]):
            excluded.append(s)
            warn("missing_base", "缺少共同起点价格，不参与股票收益对比。", s)
        else:
            bases[s] = base.iloc[0]
            series[s] = (daily.close / bases[s] - 1).tolist()
        groups = {}
        for f in fills:
            if f["symbol"] == s:
                groups.setdefault((f["date"], f["action_type"]), []).append(f)
        markers = []
        for (d, action), group in groups.items():
            q = sum(f["shares"] for f in group)
            markers.append(dict(date=d, action_type=action, fill_ids=[f["fill_id"] for f in group],
                                count=len(group), shares=q,
                                weighted_price=sum(f["price"] * f["shares"] for f in group) / q,
                                min_price=min(f["price"] for f in group), max_price=max(f["price"] for f in group),
                                realized_pnl=sum(f["realized_pnl"] for f in group) if action in ("减仓", "清仓") else None))
        stocks.append(dict(symbol=s, status="partial" if any(w.get("symbol") == s for w in warnings) else "complete",
                           daily=daily.rename_axis("date").reset_index().to_dict("records"), markers=markers))
    account = end[["total_equity"]].copy()
    account["nav"] = account.total_equity / initial_cash
    account["return_pct"] = account.nav - 1
    account["worst_drawdown"] = eq.groupby("date").worst_drawdown.min()
    _same(-account.worst_drawdown.min(), stats["max_drawdown"], "回撤与 evaluate 不一致")
    note = "按可用 Bar 聚合；首尾可能为不完整交易日，未进行交易日历完整性校验。"
    if any(metadata.get(k) and str(metadata[k])[:10] != v for k, v in (("start", days[0]), ("end", days[-1]))):
        note += " 配置日期与实际范围不同，页面使用实际账户范围。"
    return clean_json(dict(schema_version=1,
        meta=dict(run_id=metadata.get("run_id"), start=days[0], end=days[-1], start_ts=t0, end_ts=t1,
                  timezone="Asia/Shanghai", initial_cash=initial_cash, symbols=symbols,
                  price_basis="价格口径与本次回测输入一致；复权标识：" + str(metadata.get("adjustment", "未知")),
                  volume_unit=metadata.get("volume_unit", "原始单位"), aggregation_note=note),
        summary=dict(initial_cash=initial_cash, final_equity=end.total_equity.iloc[-1],
                     total_return=end.total_equity.iloc[-1] / initial_cash - 1, fill_count=len(fills),
                     sell_count=len(closed), realized_pnl=sum(pnl.values()), max_drawdown=stats["max_drawdown"]),
        account_daily=account.rename_axis("date").reset_index().to_dict("records"),
        comparison=dict(base_date=days[0], base_ts=t0, stock_base_prices=bases, series=series, excluded_symbols=excluded),
        stocks=stocks, fills=fills, warnings=warnings))
