"""自包含离线 HTML，调用时才加载 Plotly；原子替换，不开启浏览器。"""
import json
import os
from pathlib import Path
import re
import tempfile

from analytics.visualization_data import clean_json


def _figures(data):
    import plotly.graph_objects as go

    dates = [r["date"] for r in data["account_daily"]]
    def line(name, y, **kw):
        return go.Scatter(x=dates, y=y, name=name, mode="lines+markers", connectgaps=False,
                          marker=dict(size=3), **kw)

    figures = {}
    def put(key, traces, title, unit, percent=False, **layout):
        fig = go.Figure(traces)
        fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(size=15), y=.98, yanchor="top"),
            height=300, margin=dict(l=65, r=65, t=85, b=45), hovermode="closest", dragmode="zoom",
            xaxis=dict(type="date", rangeslider=dict(visible=False)),
            yaxis=dict(title=unit, tickformat=".1%" if percent else None, fixedrange=True),
            legend=dict(orientation="h", y=1.12), **layout)
        # JSON lists rather than typed arrays make the artifact easy to inspect and portable.
        figures[key] = json.loads(fig.to_json())

    put("account", [line("账户净值", [r["nav"] for r in data["account_daily"]])], "账户净值（日末）", "初始资金 = 1")
    put("drawdown", [line("当日最深回撤", [r["worst_drawdown"] for r in data["account_daily"]])], "当日最深回撤（全 Bar 峰值）", "回撤", True)
    put("comparison", [line("策略账户收益率", [r["return_pct"] for r in data["account_daily"]])] +
        [line(s + " 价格涨跌幅", values) for s, values in data["comparison"]["series"].items()],
        "固定起点收益对比", "收益率", True)
    shapes = {"买入": "triangle-up", "加仓": "cross", "减仓": "triangle-down", "清仓": "x"}
    for i, stock in enumerate(data["stocks"]):
        rows = stock["daily"]
        def values(key):
            return [r[key] for r in rows]
        traces = [go.Candlestick(x=dates, open=values("open"), high=values("high"), low=values("low"),
                    close=values("close"), name="日 K", increasing_line_color="#dc3545", decreasing_line_color="#15966a"),
                  line("收盘价", values("close"), visible=False)]
        traces += [line(f"MA{n}", values(f"ma{n}")) for n in (5, 20, 60)]
        traces += [line("账户成本（不含费）", values("cost_basis"), line=dict(shape="hv", dash="dash"))]
        hover_by_day = {}
        for action, shape in shapes.items():
            ms = [m for m in stock["markers"] if m["action_type"] == action]
            text = [f"{m['date']} {action}<br>{m['count']} 笔 / {m['shares']} 股<br>汇总成交均价 {m['weighted_price']:.6f}"
                    f"<br>最低/最高 {m['min_price']:.6f} / {m['max_price']:.6f}"
                    + (f"<br>已实现盈亏 {m['realized_pnl']:.4f}" if m['realized_pnl'] is not None else "") for m in ms]
            for m, label in zip(ms, text):
                hover_by_day.setdefault(m["date"], []).append(label)
            traces.append(go.Scatter(x=[m["date"] for m in ms], y=[m["weighted_price"] for m in ms],
                name=action, mode="markers+text", text=[action] * len(ms), textposition="top center",
                customdata=[[stock["symbol"], m["date"], action] for m in ms], hovertext=text,
                hoverinfo="text", marker=dict(symbol=shape, size=12)))
        # A price/MA trace can win Plotly's hit test when it overlaps a fill glyph.
        # Its tooltip also includes all actual fill groups for that exact date.
        for trace in traces[:6]:
            trace.hoverinfo = "text"
            def display(value):
                return "—" if value is None else str(value)
            trace.hovertext = [(f"{r['date']}<br>开 {display(r['open'])} / 高 {display(r['high'])} / 低 {display(r['low'])} / 收 {display(r['close'])}"
                               if trace.type == "candlestick" else f"{r['date']} {trace.name}: {display(trace.y[j])}")
                               + ("<br>" + "<br>".join(hover_by_day[r['date']]) if r['date'] in hover_by_day else "")
                               for j, r in enumerate(rows)]
        traces.append(go.Scatter(x=[], y=[], name="选中成交", mode="markers", marker=dict(symbol="circle-open", size=22,
                       color="#7040cd", line=dict(width=3)), hovertemplate="%{text}<extra></extra>"))
        put(f"s{i}-price", traces, "价格与实际成交", "元")
        figures[f"s{i}-price"]["layout"]["height"] = 460
        put(f"s{i}-volume", [go.Bar(x=dates, y=values("volume"), name="成交量")], "成交量", data["meta"]["volume_unit"])
        put(f"s{i}-position", [line("持仓股数", values("shares")),
            line("仓位", values("position_weight"), yaxis="y2")], "日末持仓", "股",
            yaxis2=dict(title="仓位", overlaying="y", side="right", tickformat=".1%", fixedrange=True))
    return figures


def write_backtest_html(report_data: dict, output_path) -> Path:
    """内嵌日频数据与一份 Plotly.js，临时文件成功写入后原子替换。返回绝对路径。"""
    from plotly.offline import get_plotlyjs

    root = Path(__file__).resolve().parent / "templates"
    payload = clean_json(dict(report=report_data, figures=_figures(report_data)))
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    serialized = serialized.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    page = (root / "backtest_report.html").read_text(encoding="utf-8")
    replacements = {"__PLOTLY__": get_plotlyjs(), "__DATA__": serialized,
                    "__APP__": (root / "backtest_report.js").read_text(encoding="utf-8")}
    page = re.sub(r"__PLOTLY__|__DATA__|__APP__", lambda m: replacements[m.group()], page)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
            f.write(page)
        os.replace(temp, target)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()
    return target
