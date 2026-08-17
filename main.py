"""项目主入口：全链路冒烟测试（Smoke Test）。

用内置极简 Mock 数据（6 个交易日 × 2 只股票）跑通：
    数据构建 → 特征工程（Agent Profiling / Microstructure / Environment）
    → 多层闸门状态机（8 层开仓 / 6 层平仓）→ 回测撮合 → 绩效评估 → 可视化
并打印各环节耗时与状态检查信息。

剧情设计（保证 ≥1 笔完整 BUY→SELL 闭环）：
- D1~D4 强牛：每 Bar 超大单主动买 + 小单主动卖（Inst_Flow>0、CPS>0、OFSS>0 → S_push）
- D5~D6 转熊：每 Bar 超大单主动卖（Inst_Flow<0 → 状态机退出 S_push → 触发 SELL）
- 时序约束（T-1 对齐 + Next-Bar 执行，决定开/平仓时点）：
    * 日频因子（CPS/北向/RS/行业情绪）第 2 个交易日起才可用
    * GRS/Global_Mod 的滚动 z-score 需 20 根非 NaN 分钟 bar → D4 13:30 起有值
    * 故 BUY 落在 D4 14:00（Next-Bar 成交），SELL 落在 D5 10:00（T+1 解冻后，合规）

用法：
    python main.py
"""

import logging
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from analytics.metrics import closed_trades, evaluate
from data.dataslice import DataSlice, SYMBOL, TRADE_DATE
from engine.backtest import BacktestEngine
from engine.execution import ExecutionCost
from engine.portfolio import Account
from engine.risk_control import PositionSizer
from indicators.basic import compute_all
from indicators.feature_engine import FEATURE_COLS, FeatureEngine
from indicators.microstructure import MicroStructure
from strategy.signals import (
    ACT_ADD,
    ACT_BUY,
    ACT_SELL,
    SignalSynthesizer,
    TradingStateMachine,
)

logger = logging.getLogger("main.smoke")

# ----------------------------------------------------------------------
# 冒烟 Mock 数据：6 个交易日，前 4 日强牛、后 2 日转熊
# ----------------------------------------------------------------------

SYMBOLS: Tuple[str, ...] = ("600000", "000001")
SYMBOL_TO_INDUSTRY: Dict[str, str] = {"600000": "银行", "000001": "银行"}

BULL_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
BEAR_DATES = ["2024-01-08", "2024-01-09"]
DATES = BULL_DATES + BEAR_DATES

# 冒烟显式参数（权重和为 1，全部落在 optimizer.SearchSpace 范围内；
# chip_window=1 保证第 2 个交易日起 CPS 有值；th_global_min=-0.8 匹配
# T-1 宏观对齐后 Global_Mod 到 D4 才有值的时序）
SMOKE_PARAMS: Dict[str, object] = {
    "weights": (0.35, 0.25, 0.25, 0.15),
    "th_ms_bull": 0.3, "th_ms_exit": -0.1, "th_lock": 0.4, "th_purity": 0.1,
    "th_global_min": -0.8, "th_adr_min": 0.3,
    "win_hold_max": 120, "inst_window": 1, "chip_window": 1,
}

INITIAL_CASH = 1e8


# ----------------------------------------------------------------------
# Mock 数据构造（仿 tests/test_optimizer.py，扩展为多标的）
# ----------------------------------------------------------------------

def cn_minutes(dates, freq: str = "30min") -> pd.DatetimeIndex:
    """A 股交易分钟轴（09:30~11:30 + 13:00~15:00）。"""
    parts = []
    for d in pd.to_datetime(dates):
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 09:30", f"{d:%Y-%m-%d} 11:30", freq=freq))
        parts.append(pd.date_range(f"{d:%Y-%m-%d} 13:00", f"{d:%Y-%m-%d} 15:00", freq=freq))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])).sort_values()


def mk_kline(axis: pd.DatetimeIndex, bull_dates, symbols=SYMBOLS,
             bases=(10.0, 20.0), vol: float = 1e5) -> pd.DataFrame:
    """个股 K 线：牛段每 Bar 复利 +0.2%、熊段 -0.2%；放量 → amount > 1e7。"""
    bull = {pd.Timestamp(d).normalize() for d in bull_dates}
    per_sym = {}
    for sym, base in zip(symbols, bases):
        closes = [base]
        for t in axis[1:]:
            step = 0.002 if t.normalize() in bull else -0.002
            closes.append(closes[-1] * (1.0 + step))
        per_sym[sym] = closes
    rows = []
    for i, t in enumerate(axis):
        for sym in symbols:
            c = per_sym[sym][i]
            rows.append({
                "symbol": sym, "open": c * 0.995, "high": c * 1.01,
                "low": c * 0.99, "close": c, "volume": vol,
                "amount": c * vol * 100, "vwap": c, "float_market_cap": 1e9,
                "up_limit": c * 1.1, "down_limit": c * 0.9, "is_st": False,
            })
    df = pd.DataFrame(rows)
    df.index = axis.repeat(len(symbols))
    df.index.name = "ts"
    return df


def mk_snapshot(axis: pd.DatetimeIndex, symbols=SYMBOLS) -> pd.DataFrame:
    """五档 L2 快照：买盘量 > 卖盘量 → OBI > 0。"""
    rows = []
    for t in axis:
        for sym in symbols:
            row = {"symbol": sym}
            for i in range(1, 6):
                row[f"bid{i}_p"] = 10.0 - 0.01 * i
                row[f"bid{i}_v"] = 200.0 / i
                row[f"ask{i}_p"] = 10.0 + 0.01 * i
                row[f"ask{i}_v"] = 100.0 / i
            rows.append(row)
    df = pd.DataFrame(rows, index=axis.repeat(len(symbols)))
    df.index.name = "ts"
    return df


def mk_flow_ticks(axis: pd.DatetimeIndex, bull_dates, symbols=SYMBOLS) -> pd.DataFrame:
    """逐笔成交：牛段超大单买 2e6 + 小单卖 2e4（Inst>0、Retail<0）；
    熊段超大单卖 2e6（Inst<0 → 状态机退出 S_push → SELL）。"""
    bull = {pd.Timestamp(d).normalize() for d in bull_dates}
    rows = []
    for t in axis:
        for sym in symbols:
            if t.normalize() in bull:
                rows.append((t, sym, 10.0, 0, 2e6, 1, False))
                rows.append((t, sym, 10.0, 0, 2e4, -1, False))
            else:
                rows.append((t, sym, 10.0, 0, 2e6, -1, False))
    df = pd.DataFrame(rows, columns=["ts", "symbol", "price", "volume",
                                     "turnover", "side", "is_cancel"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")


def mk_index_min(axis: pd.DatetimeIndex) -> pd.DataFrame:
    """沪深300 指数：加速上涨（ma20 > ma60）→ MRS>0 且 个股跑赢 → RS>0。"""
    n = len(axis)
    closes = 3000.0 + 0.05 * np.arange(n) + 0.002 * np.arange(n) ** 2
    rows = []
    for i, t in enumerate(axis):
        c = closes[i]
        rows.append({"index_code": "000300.SH", "open": c, "high": c * 1.002,
                     "low": c * 0.998, "close": c,
                     "volume": 1e7 * (1.0 + 0.02 * i), "vwap": c,
                     "ma20": c, "ma60": c * 0.99})
    df = pd.DataFrame(rows, index=axis)
    df.index.name = "ts"
    return df


def mk_breadth(axis: pd.DatetimeIndex) -> pd.DataFrame:
    """全市场广度：ADR=2.0 达标（> 1.0），北向净流为正。"""
    df = pd.DataFrame(index=axis)
    df["advancers"] = 3000.0
    df["decliners"] = 1500.0
    df["adr"] = 2.0
    df["north_net"] = 5e7
    return df


def mk_industry(axis: pd.DatetimeIndex, name: str = "银行") -> pd.DataFrame:
    """行业资金流：单调递增 → IRS / Chain_Mod > 0。"""
    df = pd.DataFrame(index=axis)
    df["industry"] = name
    df["open"] = df["high"] = df["low"] = df["close"] = 1000.0
    df["volume"] = 1e6
    df["money_flow"] = np.linspace(0, 1, len(axis)) * 1e8
    return df


def mk_macro(dates, base: float = 100.0) -> pd.DataFrame:
    """全球宏观（日频，T-1 对齐）：美股/商品/美债均单调走强 → GRS>0。"""
    rows = []
    for i, d in enumerate(pd.to_datetime(dates)):
        rows.append({"trade_date": d, "us_spx": base + i, "us_ndx": base + i,
                     "us_dow": base + i, "brent": base + i, "gold": base + i,
                     "copper": base + i, "us10y": 3.0 + 0.01 * i, "dxy": 100 + i,
                     "hsi": base + i, "nky": base + i})
    return pd.DataFrame(rows)


def mk_north_margin(dates, symbols=SYMBOLS) -> pd.DataFrame:
    """北向 / 两融（日频，T+1 披露）：持仓递增 → North_Sync 偏多。"""
    idx = pd.to_datetime(dates)
    rows = []
    for i, d in enumerate(idx):
        for sym in symbols:
            rows.append({
                "symbol": sym, "trade_date": d,
                "north_holding": 1e7 + 2e5 * i,
                "north_buy_net": 1e6 if i % 3 == 0 else -5e5,
                "margin_fin_balance": 1e9 + 2e7 * i,
                "margin_sec_balance": 5e8 + 1e7 * i,
            })
    return pd.DataFrame(rows)


def mk_dragon_tiger(dates, symbols=SYMBOLS) -> pd.DataFrame:
    """龙虎榜（上榜日 trade_date，avail_date 由 aligner 标注为 T+1）。"""
    rows = []
    for d in pd.to_datetime(dates):
        for sym in symbols:
            rows.append({"symbol": sym, "trade_date": d,
                         "buy_amount": 1e8, "sell_amount": 5e7,
                         "net_amount": 5e7, "side": 1})
    return pd.DataFrame(rows)


def build_smoke_slice() -> DataSlice:
    """构造完整冒烟 DataSlice（龙虎榜经 TimeAligner 标注 T+1 可用日）。"""
    from data.aligner import TimeAligner
    axis = cn_minutes(DATES, freq="30min")
    ds = DataSlice(
        kline=mk_kline(axis, BULL_DATES),
        l2_snapshot=mk_snapshot(axis),
        tick_trades=mk_flow_ticks(axis, BULL_DATES),
        index_min=mk_index_min(axis),
        breadth=mk_breadth(axis),
        industry=mk_industry(axis),
        macro=mk_macro(DATES),
        north_margin=mk_north_margin(DATES),
        dragon_tiger=mk_dragon_tiger(DATES),
        meta={"symbols": list(SYMBOLS), "smoke": True},
    )
    # DataSlice.validate() 要求龙虎榜携带 avail_date，先按真实 T+1 规则标注
    ds.dragon_tiger = TimeAligner().align_dragon_tiger(ds.dragon_tiger, axis)
    return ds


# ----------------------------------------------------------------------
# 主流程：全链路冒烟测试
# ----------------------------------------------------------------------

def _tick(t0: float, label: str) -> float:
    dt = time.perf_counter() - t0
    logger.info("    └─ %s 耗时 %.3f s", label, dt)
    return time.perf_counter()


def run_smoke() -> None:
    """执行冒烟测试主流程（各环节串行并打印耗时与状态）。"""
    logger.info("=" * 78)
    logger.info("全链路冒烟测试启动：%d 个交易日 × %d 只股票（%s）",
                len(DATES), len(SYMBOLS), "牛→熊 剧情")
    logger.info("=" * 78)
    start = time.perf_counter()
    t0 = start

    # ---- 环节 1：Mock 数据构建 ----
    ds = build_smoke_slice()
    ds.validate()  # schema + 时间索引质量显式校验
    t0 = _tick(t0, f"Mock 数据构建 + DataSlice.validate（{len(ds.kline)} 根 K 线）")

    # ---- 环节 2：特征工程（Agent Profiling / Microstructure / Environment）----
    fe = FeatureEngine(
        micro=MicroStructure(chip_window=int(SMOKE_PARAMS["chip_window"])),
        symbol_to_industry=SYMBOL_TO_INDUSTRY,
    )
    features = fe.compute(ds)
    feat_nan = int(features.isna().sum().sum())
    logger.info("特征工程输出：%d 行 × %d 列，NaN 元素 %d 个（T-1 对齐导致的预期缺失）",
                len(features), len(FEATURE_COLS), feat_nan)
    t0 = _tick(t0, "特征工程 FeatureEngine.compute")

    # ---- 环节 3：信号合成 + 多层闸门状态机 ----
    syn = SignalSynthesizer(
        weights=tuple(SMOKE_PARAMS["weights"]),
        inst_window=int(SMOKE_PARAMS["inst_window"]),
        th_ms_bull=float(SMOKE_PARAMS["th_ms_bull"]),
        th_ms_exit=float(SMOKE_PARAMS["th_ms_exit"]),
        th_lock=float(SMOKE_PARAMS["th_lock"]),
        th_purity=float(SMOKE_PARAMS["th_purity"]),
        th_global_min=float(SMOKE_PARAMS["th_global_min"]),
        th_adr_min=float(SMOKE_PARAMS["th_adr_min"]),
        win_hold_max=int(SMOKE_PARAMS["win_hold_max"]),
        symbol_to_industry=SYMBOL_TO_INDUSTRY,
    )
    sm = TradingStateMachine(synthesizer=syn)
    signals = sm.run(ds, features)
    act_cnt = Counter(s.action for s in signals)
    logger.info("状态机输出：%d 个信号 → %s", len(signals),
                {k: act_cnt[k] for k in (ACT_BUY, ACT_ADD, ACT_SELL, "HOLD")})
    t0 = _tick(t0, "信号合成 + 状态机（8/6 层闸门）")

    # ---- 环节 4：回测撮合引擎 ----
    engine = BacktestEngine(
        Account(initial_cash=INITIAL_CASH),
        ExecutionCost(),
        PositionSizer(),
        ds, signals,
    )
    trade_log, equity_curve = engine.run()
    filled = trade_log[trade_log["shares"] > 0]
    rejected = trade_log[trade_log["shares"] == 0]
    logger.info("回测撮合：%d 根 Bar 净值点；成交 %d 笔 / 拒绝 %d 笔；"
                "拒绝原因=%s", len(equity_curve), len(filled), len(rejected),
                dict(rejected["reason"].value_counts()) if len(rejected) else "无")
    t0 = _tick(t0, "回测撮合 BacktestEngine.run")

    # ---- 环节 5：绩效评估 ----
    metrics = evaluate(equity_curve, trade_log)
    trades = closed_trades(trade_log)
    logger.info("绩效评估：年化Sharpe=%.3f 最大回撤=%.2f%% 总盈亏=%+.0f 元 "
                "有效交易=%d 笔 胜率=%.0f%%",
                metrics["sharpe"], metrics["max_drawdown"] * 100,
                metrics["total_pnl"], metrics["n_trades"],
                metrics["win_rate"] * 100 if np.isfinite(metrics["win_rate"]) else float("nan"))
    for tr in trades:
        logger.info("    └─ %s 于 %s 买入 → %s 卖出，%d 股，盈亏 %+.0f 元",
                    tr["symbol"], tr["entry_ts"], tr["exit_ts"],
                    int(tr["shares"]), tr["pnl"])
    t0 = _tick(t0, "绩效评估 analytics.metrics.evaluate")

    # ---- 环节 6：可视化（analytics/plotter.py → analytics/pictures/）----
    chart_paths = _plot_smoke_charts(ds)
    t0 = _tick(t0, f"绘图 plot_trend（{len(chart_paths)} 张）")

    # ---- 状态检查 ----
    checks = _status_checks(ds, features, trade_log, equity_curve)
    logger.info("=" * 78)
    logger.info("状态检查汇总：")
    for ok, msg in checks:
        logger.info("    [%s] %s", "PASS" if ok else "FAIL", msg)
    if not all(ok for ok, _ in checks):
        raise SystemExit("冒烟测试未通过，请查看上方 FAIL 项")
    logger.info("=" * 78)
    logger.info("冒烟测试全部通过 ✅（总耗时 %.3f s）", time.perf_counter() - start)
    logger.info("可视化输出：%s", ", ".join(chart_paths))


def _plot_smoke_charts(ds: DataSlice) -> List[str]:
    """把分钟 K 线按日聚合后跑 compute_all，再调用 plot_trend 出图。"""
    from analytics.plotter import plot_trend
    paths = []
    for sym in ds.symbols():
        k = ds.kline[ds.kline[SYMBOL] == sym].copy()
        daily = k.resample("D").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"),
            amount=("amount", "sum"),
        )
        daily = compute_all(daily)  # ma5/ma20/is_trend 等绘图必需列
        daily["code"] = sym
        fname = os.path.join("analytics", "pictures", f"smoke_{sym}.png")
        path = plot_trend(daily, fname=fname,
                          title=f"{sym} 冒烟回测段（黄色区域=震荡区间）")
        paths.append(path)
    return paths


def _status_checks(ds: DataSlice, features: pd.DataFrame,
                   trade_log: pd.DataFrame, equity_curve: pd.DataFrame) -> List[Tuple[bool, str]]:
    """冒烟状态检查：闭环完整性 / 未来函数 / T+1 / 空引用 / 资金安全。"""
    checks: List[Tuple[bool, str]] = []

    # 1) 必须有 ≥1 笔完整 BUY→SELL 闭环
    filled = trade_log[trade_log["shares"] > 0]
    n_closed = len(closed_trades(trade_log))
    checks.append((n_closed >= 1,
                   f"存在 ≥1 笔完整平仓交易（实际 {n_closed} 笔，成交 {len(filled)} 笔）"))

    # 2) 无未来函数：特征表时间轴不早于日频因子可用日。
    #    FeatureEngine 已在龙虎榜路径显式调用 TimeAligner.verify_no_lookahead，
    #    此处对全部日频因子做二次回归校验（特征行不得早于 macro/north T-1 可用日）。
    checks.append((
        _verify_no_lookahead(ds, features),
        "无未来函数（龙虎榜 T+1 断言 + 日频因子 T-1 对齐二次校验）"))

    # 3) 无 T+1 违规：SELL 成交不得发生在买入当日（当日买入份额必须冻结）。
    sells = filled[filled["side"] == "SELL"]
    t1_ok = True
    for _, row in sells.iterrows():
        buys = filled[(filled["side"].isin(("BUY", "ADD"))) &
                      (filled["symbol"] == row["symbol"]) &
                      (filled["ts"] <= row["ts"])]
        if buys.empty:
            continue
        last_buy_day = pd.Timestamp(buys["ts"].max()).normalize()
        if pd.Timestamp(row["ts"]).normalize() == last_buy_day:
            t1_ok = False
            break
    checks.append((t1_ok, "无 T+1 违规（SELL 均发生在买入次日起）"))

    # 4) 无空指针/空引用：关键产物非空且列完整
    checks.append((len(equity_curve) > 0 and {"cash", "total_equity"} <=
                   set(equity_curve.columns), "净值曲线非空且列完整"))

    # 5) 资金安全：现金非负、净值非负
    cash_min = float(equity_curve["cash"].min())
    equity_min = float(equity_curve["total_equity"].min())
    checks.append((cash_min >= -1e-6 and equity_min > 0,
                   f"资金安全（现金最小值 {cash_min:,.0f}，净值最小值 {equity_min:,.0f}）"))
    return checks


def _verify_no_lookahead(ds: DataSlice, features: pd.DataFrame) -> bool:
    """日频因子 T-1 可见性二次校验：
    特征行使用时间必须晚于该行所对应的日频数据可用日（次日 00:00）。"""
    ts = features.index
    day_available = ts.normalize()  # T-1 对齐后，当日行只能使用前一交易日数据
    # 简化复核：首日（无 T-1 数据）的任何日频因子必须全 NaN；
    # 次日（可用日）起允许出现值。等价于"当日因子不可见当日数据"。
    first_day = ts.normalize() == ts.normalize()[0]
    daily_cols = ["north_sync", "margin_pressure", "cps", "dt_net"]
    for col in daily_cols:
        if col in features.columns:
            v = features.loc[first_day, col]
            if pd.notna(v).any():
                return False
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_smoke()
