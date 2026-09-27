"""日频展示短测试；全部为合成数据，不获取行情或寻优。"""
import copy

import numpy as np
import pandas as pd
import pytest

from analytics.visualization_data import build_visualization_data


def inputs(n=4, count=4):
    axis = pd.bdate_range('2024-01-02 15:00', periods=count)
    symbols = [f'{i:06d}' for i in range(n)]
    bars, positions = [], []
    for j, ts in enumerate(axis):
        for s in symbols:
            bars.append(dict(ts=ts, symbol=s, open=10, high=16, low=9, close=11+j*.001, volume=100))
            positions.append(dict(ts=ts, symbol=s, shares=0, mark_price=None, market_value=0, total_equity=100000))
    return dict(kline=pd.DataFrame(bars), trade_log=pd.DataFrame(),
        equity_curve=pd.DataFrame(dict(ts=axis, total_equity=100000, position_value=0)),
        positions_daily=pd.DataFrame(positions), symbols=symbols, initial_cash=100000, metadata={})


def ledger():
    data=inputs(1)
    rows=[]
    for i,(side,q,p,fee) in enumerate([('BUY',1000,10,10),('ADD',500,14,5),('SELL',600,15,9),('SELL',900,12,10)]):
        rows.append(dict(ts=data['equity_curve'].ts.iloc[i],symbol='000000',side=side,shares=q,price=p,
                         amount=q*p,commission=fee,transfer_fee=0,stamp_duty=0,fill_id=f'f{i}',event_seq=i))
    data['trade_log']=pd.DataFrame(rows)
    data['positions_daily']['shares']=[1000,1500,900,0]
    data['positions_daily']['mark_price']=12.
    data['positions_daily']['market_value']=data['positions_daily'].shares*12
    data['equity_curve']['position_value']=data['positions_daily'].market_value
    return data


def test_cost_pnl_and_immutability():
    data=ledger(); before=copy.deepcopy(data)
    out=build_visualization_data(**data)
    assert [f['action_type'] for f in out['fills']]==['买入','加仓','减仓','清仓']
    assert [f['realized_pnl'] for f in out['fills']]==[None,None,2185.,581.]
    assert out['summary']['realized_pnl']==2766
    assert [r['cost_basis'] for r in out['stocks'][0]['daily']]==[10,17000/1500,17000/1500,None]
    for key in ('kline','trade_log','equity_curve','positions_daily'):pd.testing.assert_frame_equal(data[key],before[key])


def test_aggregation_timezone_gaps_and_base():
    data=inputs(2,3)
    bars=data['kline']; t=bars.ts.iloc[0]
    bars.loc[0,['open','high','low','close','volume']]=[10,12,9,11,100]
    second=bars.iloc[[0]].copy();second['ts']=t+pd.Timedelta(minutes=1)
    second[['open','high','low','close','volume']]=[11,13,10,12,200]
    # Account end of first day is the second bar; exact t0 remains original first bar.
    extra=data['equity_curve'].iloc[[0]].copy();extra['ts']=second.ts.iloc[0]
    data['equity_curve']=pd.concat([data['equity_curve'],extra])
    data['positions_daily'].loc[data['positions_daily'].ts.eq(t),'ts']=second.ts.iloc[0]
    bars=bars.drop(index=2) # A no market D2
    bars.loc[1,'ts']=t+pd.Timedelta(seconds=30) # B misses exact common base
    data['kline']=pd.concat([bars,second])
    out=build_visualization_data(**data); a=out['stocks'][0]['daily']
    assert [a[0][k] for k in ['open','high','low','close','volume']]==[10,13,9,12,300]
    assert a[1]['close'] is None and a[1]['ma5'] is None
    assert out['comparison']['series']['000000'][0]==pytest.approx(.2)
    assert out['comparison']['excluded_symbols']==['000001']
    data['kline']['ts']=data['kline'].ts.dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC')
    assert build_visualization_data(**data)==out


def test_missing_volume_and_explicit_empty_market():
    data=inputs(1,1);data['kline']['volume']=np.nan
    assert build_visualization_data(**data)['stocks'][0]['daily'][0]['volume'] is None
    data['kline']=pd.DataFrame()
    out=build_visualization_data(**data)
    assert out['warnings'] and out['stocks'][0]['daily'][0]['close'] is None


@pytest.mark.parametrize('kind', ['cash','equity','duplicate_equity','price','amount','side','time','id','seq','snapshot','missing','value','shares','duplicate_bar','ohlc'])
def test_invalid_inputs(kind):
    d=ledger()
    if kind=='cash':d['initial_cash']=None
    elif kind=='equity':d['equity_curve'].loc[0,'total_equity']=0
    elif kind=='duplicate_equity':d['equity_curve']=pd.concat([d['equity_curve'],d['equity_curve'].iloc[[0]]])
    elif kind=='price':d['trade_log'].loc[0,'price']=0
    elif kind=='amount':d['trade_log'].loc[0,'amount']+=1
    elif kind=='side':d['trade_log'].loc[0,'side']='OTHER'
    elif kind=='time':d['trade_log'].loc[0,'ts']-=pd.Timedelta(days=10)
    elif kind=='id':d['trade_log'].loc[1,'fill_id']='f0'
    elif kind=='seq':d['trade_log'].loc[1,'event_seq']=0
    elif kind=='snapshot':d['positions_daily'].loc[0,'ts']-=pd.Timedelta(minutes=1)
    elif kind=='missing':d['positions_daily']=d['positions_daily'].iloc[1:]
    elif kind=='value':d['equity_curve'].loc[0,'position_value']+=1
    elif kind=='shares':d['positions_daily'].loc[0,'shares']+=1
    elif kind=='duplicate_bar':d['kline']=pd.concat([d['kline'],d['kline'].iloc[[0]]])
    elif kind=='ohlc':d['kline'].loc[0,'high']=1
    with pytest.raises(ValueError):build_visualization_data(**d)


def test_intraday_drawdown_initial_return_and_clipping():
    d=inputs(1,2);ts=d['equity_curve'].ts.iloc[0]
    d['initial_cash']=1000
    d['equity_curve']=pd.DataFrame(dict(ts=[ts,ts+pd.Timedelta(minutes=1),ts+pd.Timedelta(minutes=2),d['equity_curve'].ts.iloc[-1]],total_equity=[1200,900,1100,990],position_value=0))
    d['positions_daily']['total_equity']=[1100,990]
    d['positions_daily'].loc[0,'ts']=ts+pd.Timedelta(minutes=2)
    out=build_visualization_data(**d)
    assert out['summary']['max_drawdown']==.25
    assert out['account_daily'][0]['worst_drawdown']==-.25
    assert out['account_daily'][1]['return_pct']==pytest.approx(-.01)
    extra=d['kline'].iloc[[0]].copy();extra['ts']=ts-pd.Timedelta(minutes=1);extra['high']=999
    d['kline']=pd.concat([d['kline'],extra])
    assert build_visualization_data(**d)==out


def test_legacy_rejections_pnl_keys_and_missing_pnl(monkeypatch):
    from analytics import metrics
    d=ledger();log=d['trade_log'].drop(columns=['fill_id','event_seq'])
    rejected=log.iloc[[0]].copy();rejected['shares']=0;rejected['amount']=0
    d['trade_log']=pd.concat([rejected,log.iloc[:2],rejected,log.iloc[2:]],ignore_index=True)
    out=build_visualization_data(**d)
    assert {f['fill_id'] for f in out['fills'] if f['side']=='SELL'}=={r['sell_fill_id'] for r in metrics.closed_trades(d['trade_log'])}
    original=metrics.closed_trades
    monkeypatch.setattr(metrics,'closed_trades',lambda log: original(log)[:1])
    with pytest.raises(ValueError,match='sell_fill_id'):build_visualization_data(**d)


def test_ma_valid_days_and_grouped_actual_prices():
    d=inputs(1,7);d['kline']=d['kline'].drop(index=3)
    out=build_visualization_data(**d);rows=out['stocks'][0]['daily']
    assert rows[4]['ma5'] is None
    assert rows[5]['ma5']==pytest.approx(np.mean([11,11.001,11.002,11.004,11.005]))
    d=ledger();r=d['trade_log'].iloc[0].copy();r['shares']=400;r['amount']=4000;r['fill_id']='extra';r['event_seq']=9
    d['trade_log'].loc[0,['shares','amount']]=[600,6000]
    d['trade_log']=pd.concat([d['trade_log'],pd.DataFrame([r])])
    out=build_visualization_data(**d)
    assert out['fills'][1]['action_type']=='加仓'
    assert out['fills'][1]['ts']==out['fills'][0]['ts']


def test_held_market_gap_and_missing_requested_stock():
    d=ledger();d['kline']=d['kline'].drop(index=1)
    out=build_visualization_data(**d);r=out['stocks'][0]['daily'][1]
    assert r['close'] is None and r['shares']==1500 and r['market_value']==18000
    assert r['position_weight']==.18 and out['comparison']['series']['000000'][1] is None
    d=inputs(2,3);d['kline']=d['kline'][d['kline'].symbol.eq('000000')]
    out=build_visualization_data(**d)
    assert len(out['stocks'])==2 and out['stocks'][1]['status']=='partial'


def test_share_swap_and_unknown_positions():
    d=inputs(2,1)
    for s in ['000000','OUTSIDE']:
        p=d['positions_daily'].copy();p.loc[1,'symbol']=s
        with pytest.raises(ValueError):build_visualization_data(**dict(d,positions_daily=p))
    d=ledger();d['trade_log'].loc[0,'side']='SELL'
    with pytest.raises(ValueError,match='超卖'):build_visualization_data(**d)


@pytest.mark.parametrize('mode',['duplicate','extra'])
def test_pnl_identity_validation(monkeypatch,mode):
    from analytics import metrics
    d=ledger();original=metrics.closed_trades
    def corrupt(log):
        rows=original(log)
        if mode=='duplicate':rows.append(rows[0])
        else:rows[0]['sell_fill_id']='unknown'
        return rows
    monkeypatch.setattr(metrics,'closed_trades',corrupt)
    with pytest.raises(ValueError,match='sell_fill_id'):build_visualization_data(**d)


def test_same_day_weighted_group_and_sell_reopen():
    d=ledger()
    # First 1000 shares split across two buys, then a second ADD at a distinct price.
    first=d['trade_log'].iloc[0].copy();first['shares']=400;first['amount']=4000
    first['fill_id']='extra';first['event_seq']=10
    third=first.copy();third['price']=12;third['shares']=100;third['amount']=1200;third['fill_id']='extra2';third['event_seq']=11
    d['trade_log'].loc[0,['shares','amount']]=[500,5000]
    d['trade_log']=pd.concat([d['trade_log'],pd.DataFrame([first,third])])
    out=build_visualization_data(**d)
    add=next(m for m in out['stocks'][0]['markers'] if m['action_type']=='加仓' and m['count']==2)
    assert add['shares']==500 and add['weighted_price']==10.4
    assert add['fill_ids']==['extra','extra2']
