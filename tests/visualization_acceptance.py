"""可重复的合成报告与 Edge 离线交互验收（不被 pytest 自动收集）。

运行：python tests/visualization_acceptance.py
可选依赖 playwright；使用系统 Edge，无需下载浏览器。产物保留在 reports 独立目录。
"""
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly

from analytics.html_report import write_backtest_html
from analytics.visualization_data import build_visualization_data


def synthetic(n):
    dates = pd.bdate_range('2024-01-02 15:00', periods=500, normalize=False)
    symbols = [f'{i:06d}' for i in range(n)]
    bars, positions, trades, curve = [], [], [], []
    qty = {s: 0 for s in symbols}
    cash = 1_000_000.
    seq = 0
    for day, ts in enumerate(dates):
        marks = {s: 10+i*2+np.sin(day/20)+day*.002 for i,s in enumerate(symbols)}
        # Same-timestamp ordered fills, >50 records on one day, plus a same-day sell/reopen.
        orders = []
        if day == 0:
            orders = [(symbols[0], 'BUY', 100)]*55
        elif day == 1:
            orders = [(symbols[0], 'SELL', 5500), (symbols[0], 'BUY', 100)]
        elif day == 2:
            orders = [(symbols[0], 'SELL', 50)]
        elif day == 3:
            orders = [(symbols[0], 'SELL', 50)]
        elif day == 490:
            orders = [(symbols[0], 'BUY', 100)]
        for symbol, side, q in orders:
            price = marks[symbol]+.012345
            amount = q*price
            fee = 1.
            cash += -amount-fee if side == 'BUY' else amount-fee
            qty[symbol] += q if side == 'BUY' else -q
            trades.append(dict(ts=ts,symbol=symbol,side=side,shares=q,price=price,amount=amount,
                               commission=fee,stamp_duty=0,transfer_fee=0,event_seq=seq,fill_id=f'fill-{seq}'))
            seq += 1
        mv = sum(qty[s]*marks[s] for s in symbols)
        equity = cash+mv
        curve.append(dict(ts=ts,total_equity=equity,position_value=mv))
        for symbol in symbols:
            price=marks[symbol]
            if not (symbol == symbols[1] and day in (20,21)):
                bars.append(dict(ts=ts,symbol=symbol,open=price-.1,high=price+.2,low=price-.2,
                                 close=price,volume=10000+day*3))
            positions.append(dict(ts=ts,symbol=symbol,shares=qty[symbol],mark_price=price,
                                  market_value=qty[symbol]*price,total_equity=equity))
    return dict(kline=pd.DataFrame(bars),trade_log=pd.DataFrame(trades),equity_curve=pd.DataFrame(curve),
                positions_daily=pd.DataFrame(positions),symbols=symbols,initial_cash=1_000_000,
                metadata=dict(volume_unit='股', adjustment='合成数据（测试 < & " </script>）'))


def generate(n):
    start=time.perf_counter()
    data=build_visualization_data(**synthetic(n))
    folder=ROOT/'analytics'/'reports'/f'synthetic_{n}stocks_500days_{uuid4().hex[:8]}'
    path=write_backtest_html(data,folder/'backtest.html')
    info=dict(path=str(path),generation_seconds=round(time.perf_counter()-start,3),bytes=path.stat().st_size,
              stocks=n,days=500,plotly=plotly.__version__)
    print(json.dumps(info,ensure_ascii=False),flush=True)
    return path,info


def browser_check(path, n):
    from playwright.sync_api import sync_playwright
    copied=path.parent/'离线 复制'/'backtest.html'
    copied.parent.mkdir();shutil.copy2(path,copied)
    errors,requests=[],[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        context=browser.new_context(offline=True,viewport=dict(width=1440,height=1000))
        page=context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
        start=time.perf_counter();page.goto(copied.as_uri());page.wait_for_selector('body[data-ready="true"]')
        ready=round(time.perf_counter()-start,3)
        assert page.locator('.stock').count()==n
        page.evaluate('backtestReport.ensureStock(0)')
        page.locator('[data-mode="s0"]').click()
        assert page.evaluate("document.getElementById('s0-price').data[1].visible") is True
        # Actual legend click hides/restores MA and cost traces.
        price=page.locator('#s0-price')
        # Use exact legend text for stable toggles.
        for label,trace in [('MA5',2),('账户成本（不含费）',5)]:
            legend=price.locator('.legend .traces').filter(has_text=label)
            before=page.evaluate(f"document.getElementById('s0-price').data[{trace}].visible")
            legend.click();page.wait_for_timeout(350)
            after=page.evaluate(f"document.getElementById('s0-price').data[{trace}].visible")
            assert before!=after
        page.locator('#start').fill('2024-02-01');page.locator('#end').fill('2024-02-01');page.locator('#apply').click()
        page.wait_for_function("backtestReport.state.globalRange[0]==='2024-02-01'")
        assert page.evaluate("document.getElementById('s0-price').layout.xaxis.range[0] !== document.getElementById('s0-price').layout.xaxis.range[1]")
        old=page.evaluate('backtestReport.state.globalRange')
        page.locator('#start').fill('2024-03-01');page.locator('#apply').click()
        page.wait_for_function("document.getElementById('error').textContent.length>0")
        assert page.evaluate('backtestReport.state.globalRange')==old
        page.locator('#reset').click()
        page.wait_for_function("backtestReport.state.globalRange[0]==='2024-01-02'")
        page.locator('#start').fill('2030-01-01');page.locator('#end').fill('2030-02-01');page.locator('#apply').click()
        page.wait_for_function("backtestReport.state.globalRange[0]==='2025-12-01'")
        assert page.evaluate('backtestReport.state.globalRange')==['2025-12-01','2025-12-01']
        page.locator('#reset').click()
        page.wait_for_function("backtestReport.state.globalRange[0]==='2024-01-02'")
        # Actual mouse box zoom exercises Plotly relayout, not just application methods.
        price.scroll_into_view_if_needed();box=price.bounding_box()
        page.mouse.move(box['x']+180,box['y']+170);page.mouse.down()
        page.mouse.move(box['x']+430,box['y']+290,steps=12);page.mouse.up()
        page.wait_for_function('backtestReport.state.lastInteractedRange[0] !== "2024-01-02"')
        page.locator('#link').uncheck();page.evaluate('backtestReport.ensureStock(1)')
        # Programmatic Plotly relayout raises the same browser events as drag; verify source state.
        page.evaluate("Plotly.relayout('s0-price', {'xaxis.range':['2024-03-01','2024-04-01']})")
        page.wait_for_function("backtestReport.state.localRange.s0[0]==='2024-03-01'")
        page.evaluate("Plotly.relayout('s1-price', {'xaxis.range':['2024-06-03','2024-07-01']})")
        page.wait_for_function("backtestReport.state.lastInteractedRange[0]==='2024-06-03'")
        assert page.evaluate('backtestReport.state.localRange.s0[0]')=='2024-03-01'
        page.locator('#link').check()
        page.wait_for_function("backtestReport.state.globalRange[0]==='2024-06-03'")
        # Global buttons act on all charts even while unlinked.
        page.locator('#link').uncheck();page.locator('#reset').click()
        page.wait_for_function("backtestReport.state.localRange.s1[0]==='2024-01-02'")
        page.locator('#filter-action').select_option('买入')
        # Real click on SELL marker: reset old action filter, all same-day fills remain accessible.
        page.evaluate("backtestReport.rangeTo(['2024-01-02','2024-01-08'], 's0')")
        price.scroll_into_view_if_needed()
        # Obtain the rendered SELL SVG node using Plotly's trace UID.
        uid=page.evaluate("document.getElementById('s0-price')._fullData[9].uid")
        point=price.locator('.trace'+uid+' path.point').first
        point.hover(force=True)
        page.wait_for_timeout(500)
        page.screenshot(path=str(path.parent/'before-marker.png'))
        point.click(force=True)
        page.wait_for_function("document.getElementById('filter-action').value==='' && document.getElementById('filter-start').value==='2024-01-03'")
        assert page.locator('#rows tr').count()==2
        page.locator('#rows tr').first.click()
        page.wait_for_function("backtestReport.state.selectedFillId==='fill-55'")
        assert page.evaluate("document.getElementById('s0-price').data[10].y[0]")==page.evaluate("JSON.parse(document.getElementById('payload').textContent).report.fills[55].price")
        page.locator('[data-mode="s0"]').click()
        assert page.evaluate('backtestReport.state.selectedFillId')=='fill-55'
        # A user can hide the selection trace through the legend. A new table
        # selection must make it visible again, including after legend isolation.
        price.locator('.legend .traces').filter(has_text='选中成交').click()
        page.wait_for_function("document.getElementById('s0-price').data[10].visible==='legendonly'")
        page.locator('#rows tr').first.click()
        page.wait_for_function("document.getElementById('s0-price').data[10].visible===true")
        page.evaluate("backtestReport.showDay('000000','2024-01-02','加仓')")
        assert page.locator('#rows tr').count()==50
        assert page.evaluate('backtestReport.state.selectedFillId') is None
        page.locator('#rows tr').first.click()
        page.locator('#next').click()
        assert page.locator('#rows tr').count()==5
        assert page.evaluate('backtestReport.state.selectedFillId')=='fill-0'
        # Select an out-of-window fill using a real table row: only its stock moves while unlinked.
        other_range=page.evaluate('backtestReport.state.localRange.s1')
        page.locator('#clear').click();page.locator('#next').click()
        page.locator('#rows tr[data-fill="fill-59"]').click()
        page.wait_for_function("backtestReport.state.selectedFillId==='fill-59'")
        assert page.evaluate('backtestReport.state.localRange.s0[0]')>'2025-01-01'
        assert page.evaluate('backtestReport.state.localRange.s1')==other_range
        page.locator('#link').check()
        page.wait_for_function("backtestReport.state.globalRange[0]>'2025-01-01'")
        # A null market day is explicitly present and Plotly is forbidden to bridge it.
        assert page.evaluate("document.getElementById('s1-price').data[1].y[20]===null && document.getElementById('s1-price').data[1].connectgaps===false")
        # Lazy stock inherits current range, including empty market windows.
        page.evaluate(f'backtestReport.ensureStock({n-1})')
        page.set_viewport_size(dict(width=1100,height=850));page.wait_for_timeout(300)
        page.locator('#reset').click();page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(path.parent/'browser.png'),full_page=False)
        assert not errors,errors
        assert not requests,requests
        result=dict(browser=browser.version,initial_ready_seconds=ready,external_requests=requests,
                    js_errors=errors,offline_copy=str(copied),interaction='passed')
        browser.close()
    return result


if __name__=='__main__':
    results=[]
    for n in (4,20):
        path,info=generate(n)
        try:
            info.update(browser_check(path,n))
        except Exception as exc:
            info.update(interaction='failed',error=str(exc))
            (path.parent/'acceptance.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
            raise
        (path.parent/'acceptance.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
        results.append(info)
        print(json.dumps(info,ensure_ascii=False),flush=True)
