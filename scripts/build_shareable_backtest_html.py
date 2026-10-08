"""Build a single-file, offline HTML report from the three-stock backtest artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from datetime import datetime
from pathlib import Path


DEFAULT_REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "analytics"
    / "reports"
    / "research_3stocks_2023_2024_nonst_20260929"
)
STOCK_NAMES = {
    "600171": "上海贝岭",
    "600732": "爱旭股份",
    "600888": "新疆众和",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def money(value: float) -> str:
    return f"{value:,.2f} 元"


def pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def build(report_dir: Path) -> Path:
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    with (report_dir / "trade_log.csv").open(encoding="utf-8-sig", newline="") as stream:
        fills = [row for row in csv.DictReader(stream) if row["fill_id"]]
    if len(fills) != summary["filled_orders"]:
        raise ValueError("Filled order count does not match summary.json")
    chart_data = base64.b64encode((report_dir / "equity_curve.png").read_bytes()).decode("ascii")

    year_rows = []
    for year, values in sorted(summary["by_year"].items()):
        year_rows.append(
            "<tr>"
            f"<td>{esc(year)}</td>"
            f"<td class='num'>{pct(values['return'])}</td>"
            f"<td class='num'>{money(values['end_equity'])}</td>"
            f"<td class='num'>{values['sharpe']:.2f}</td>"
            f"<td class='num'>{values['max_drawdown'] * 100:.2f}%</td>"
            f"<td class='num'>{values['filled_orders']}</td>"
            f"<td class='num'>{values['closed_trades']}</td>"
            "</tr>"
        )

    fill_rows = []
    for fill in fills:
        symbol = fill["symbol"]
        fee = sum(float(fill[key]) for key in ("commission", "stamp_duty", "transfer_fee"))
        fill_rows.append(
            "<tr>"
            f"<td>{esc(fill['ts'])}</td>"
            f"<td>{esc(symbol)} {esc(STOCK_NAMES.get(symbol, ''))}</td>"
            f"<td><span class='side {esc(fill['side'].lower())}'>{esc(fill['side'])}</span></td>"
            f"<td class='num'>{int(float(fill['shares'])):,}</td>"
            f"<td class='num'>{float(fill['price']):.3f}</td>"
            f"<td class='num'>{float(fill['amount']):,.2f}</td>"
            f"<td class='num'>{fee:,.2f}</td>"
            "</tr>"
        )

    assumptions = summary["assumption"]
    stock_list = "、".join(
        f"{esc(symbol)} {esc(STOCK_NAMES.get(symbol, ''))}"
        for symbol in assumptions["symbols"]
    )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>三股回测报告 | 2023–2024</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, -apple-system, "Microsoft YaHei", sans-serif; color: #17212c; background: #f4f6f8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; line-height: 1.55; }}
    main {{ max-width: 1140px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ font-size: clamp(26px, 4vw, 38px); margin: 0 0 8px; }}
    h2 {{ font-size: 21px; margin: 0 0 16px; }}
    p {{ margin: 0 0 12px; }}
    .subtitle {{ color: #536273; }}
    .notice {{ margin: 24px 0; padding: 16px 20px; border-left: 4px solid #c17818; border-radius: 8px; background: #fff6e8; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card, section {{ background: #fff; border: 1px solid #e0e5ea; border-radius: 12px; box-shadow: 0 2px 8px #17212c08; }}
    .card {{ padding: 16px; }}
    .card .label {{ color: #5d6b78; font-size: 13px; }}
    .card .value {{ margin-top: 5px; font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }}
    section {{ margin: 18px 0; padding: 22px; }}
    figure {{ margin: 0; }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption, .footnote {{ margin-top: 8px; color: #637180; font-size: 13px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; white-space: nowrap; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e8edf1; text-align: left; }}
    th {{ background: #f7f9fb; color: #485768; }}
    tr:last-child td {{ border-bottom: 0; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .side {{ font-weight: 700; }}
    .buy {{ color: #b54827; }}
    .sell {{ color: #147d61; }}
    footer {{ margin-top: 26px; color: #637180; font-size: 13px; }}
    @media print {{ body {{ background: #fff; }} section, .card {{ box-shadow: none; break-inside: avoid; }} main {{ padding: 0; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>三股回测报告</h1>
    <p class="subtitle">{esc(assumptions['start'])} 至 {esc(assumptions['end'])} · {stock_list}</p>
  </header>
  <div class="notice">
    <strong>研究性回测，结果受数据假设影响。</strong>
    <p>这次按你的要求暂时忽略历史 ST 数据，假设三只股票在整个区间均非 ST。该假设尚未核实，可能影响买入信号和结果。</p>
    <p>独立市场指数、市场广度及流通股本数据缺失。策略参数曾参考 2024 年单只股票的结果，存在样本选择偏差。</p>
  </div>
  <div class="cards">
    <div class="card"><div class="label">期初资金</div><div class="value">{money(summary['initial_cash'])}</div></div>
    <div class="card"><div class="label">期末权益</div><div class="value">{money(summary['final_equity'])}</div></div>
    <div class="card"><div class="label">两年累计收益</div><div class="value">{pct(summary['total_return'])}</div></div>
    <div class="card"><div class="label">最大回撤</div><div class="value">{summary['metrics']['max_drawdown'] * 100:.2f}%</div></div>
    <div class="card"><div class="label">夏普比率</div><div class="value">{summary['metrics']['sharpe']:.2f}</div></div>
    <div class="card"><div class="label">平仓胜率</div><div class="value">{summary['metrics']['win_rate'] * 100:.2f}%</div></div>
  </div>
  <section>
    <h2>权益曲线</h2>
    <figure>
      <img src="data:image/png;base64,{chart_data}" alt="2023 至 2024 年三股回测权益曲线">
      <figcaption>两年连续账户回测；期末权益包含尚未平仓的持仓市值。</figcaption>
    </figure>
  </section>
  <section>
    <h2>年度结果</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>年份</th><th class="num">收益率</th><th class="num">年末权益</th><th class="num">夏普比率</th><th class="num">年内最大回撤</th><th class="num">成交笔数</th><th class="num">平仓交易</th></tr></thead>
      <tbody>{''.join(year_rows)}</tbody>
    </table></div>
    <p class="footnote">两年合计成交 {summary['filled_orders']} 笔、平仓交易 {summary['closed_trades']} 笔。年度收益按各年期初账户权益计算。</p>
  </section>
  <section>
    <h2>成交明细</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>成交时间</th><th>股票</th><th>方向</th><th class="num">股数</th><th class="num">成交价</th><th class="num">成交额（元）</th><th class="num">费用（元）</th></tr></thead>
      <tbody>{''.join(fill_rows)}</tbody>
    </table></div>
    <p class="footnote">费用为佣金、印花税与过户费之和；价格与金额是回测模型的模拟成交值。</p>
  </section>
  <footer>报告生成于 {esc(generated_at)}。本文件的图表和样式已内嵌，可直接离线查看。仅供研究，不构成投资建议。</footer>
</main>
</body>
</html>
"""
    output = report_dir / "backtest.html"
    output.write_text(document, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", nargs="?", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    print(build(args.report_dir.resolve()))
