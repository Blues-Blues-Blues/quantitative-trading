"""离线导出和小型引擎接线；禁止优化和自动开窗。"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from analytics.html_report import write_backtest_html
from analytics.visualization_data import build_visualization_data
from test_visualization_data import inputs
from test_analysis_reporting import small_engine, writer_at, strict_json


@pytest.fixture(autouse=True)
def no_optimize_or_browser(monkeypatch):
    import optuna
    import webbrowser
    import socket
    monkeypatch.setattr(optuna.study.Study,'optimize',lambda *a,**k:pytest.fail('禁止参数寻优'))
    monkeypatch.setattr(webbrowser,'open',lambda *a,**k:pytest.fail('禁止自动开窗'))
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('禁止真实网络请求'))


def test_offline_html_and_safe_text(tmp_path):
    d=inputs(1,1);d['metadata']['adjustment']='中文 "\n<&</script>'
    out=build_visualization_data(**d)
    path=write_backtest_html(out,tmp_path/'中文 空格'/'backtest.html')
    page=path.read_text(encoding='utf8')
    assert path.is_absolute() and path.stat().st_size>1_000_000
    assert page.count('plotly.js v')==1
    assert '<script src=' not in page and '<link ' not in page
    payload=page.split('<script id="payload" type="application/json">')[1].split('</script>')[0]
    assert '</script>' not in payload
    assert json.loads(payload)['report']==out
    assert len(json.loads(payload)['figures'])==6


@pytest.mark.parametrize('scenario',['cycle','open','flat'])
def test_engine_html_and_regression(tmp_path,scenario):
    baseline=small_engine(scenario=scenario);log0,curve0=baseline.run()
    w=writer_at(tmp_path);e=small_engine(w.snapshot,scenario);log,curve=e.run()
    quality=w.finalize(log,e.generated_signals,curve,kline=e.data.kline)
    pd.testing.assert_frame_equal(log,log0);pd.testing.assert_frame_equal(curve,curve0)
    assert isinstance(quality,dict) and (w.path/'backtest.html').exists()
    assert strict_json(w.path/'manifest.json')['visualization']['status']=='complete'
    assert len(w._position_days)==2 and not w._positions


def test_write_failure_and_skipped(tmp_path,monkeypatch):
    import analytics.html_report as html
    w=writer_at(tmp_path);e=small_engine(w.snapshot);log,curve=e.run()
    w.finalize(log,e.generated_signals,curve)
    assert w.manifest['visualization']['status']=='skipped'
    w=writer_at(tmp_path);e=small_engine(w.snapshot);log,curve=e.run()
    monkeypatch.setattr(html.os,'replace',lambda *a:(_ for _ in ()).throw(OSError('write failed')))
    with pytest.raises(OSError):w.finalize(log,e.generated_signals,curve,kline=e.data.kline)
    m=strict_json(w.path/'manifest.json')
    assert m['status']=='failed' and m['visualization']['status']=='failed'
    assert not (w.path/'backtest.html').exists() and not list(w.path.glob('*.tmp'))
    assert (w.path/'quality.json').exists()


def test_project_relative_path_and_lazy_import(tmp_path,monkeypatch):
    # Constructor mocked mkdir/write to avoid writing test reports outside pytest tmp.
    from analytics.reporting import AnalysisReportWriter
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path,'mkdir',lambda *a,**k:None)
    monkeypatch.setattr(AnalysisReportWriter,'_write_json',lambda *a,**k:None)
    a=AnalysisReportWriter();b=AnalysisReportWriter()
    assert a.path.parent==Path(__file__).resolve().parents[1]/'analytics'/'reports'
    assert a.path!=b.path
    result=subprocess.run([sys.executable,'-c',"import main, engine.backtest, analytics.html_report; import sys; main._plot_smoke_charts=lambda ds:[]; main.run_smoke(analysis_report=False); assert 'plotly' not in sys.modules"],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    monkeypatch.undo()  # Restore cwd before the workspace's tmp_path finalizer removes it.


def test_main_forwards_market_and_requested_symbols(tmp_path,monkeypatch):
    import main
    from analytics.reporting import AnalysisReportWriter
    original=AnalysisReportWriter.finalize
    seen=[]
    def finalize(self,log,signals,curve,*,kline=None):
        assert kline is not None
        seen.append(self)
        return original(self,log,signals,curve,kline=kline)
    monkeypatch.setattr(AnalysisReportWriter,'finalize',finalize)
    monkeypatch.setattr(main,'_plot_smoke_charts',lambda ds:[])
    ds=main.build_smoke_slice()
    ds.meta['symbols']=['MISSING']+ds.symbols()[::-1]
    monkeypatch.setattr(main,'build_smoke_slice',lambda:ds)
    # Existing synthetic smoke fixture, no market download, optimization or real backtest.
    main.run_smoke(report_dir=str(tmp_path))
    assert len(seen)==1 and (seen[0].path/'backtest.html').exists()
    assert seen[0].manifest['symbols']==ds.meta['symbols']
    assert seen[0].manifest['visualization']['status']=='partial'


@pytest.mark.parametrize('mode',['missing','stale','empty'])
def test_writer_snapshot_completeness(tmp_path,mode):
    w=writer_at(tmp_path)
    def sink(ts,positions,equity,factors):
        if mode=='missing' and ts.day==3:return
        if mode=='stale' and ts==pd.Timestamp('2024-01-03 10:01'):return
        w.snapshot(ts,positions,equity,factors)
    e=small_engine(sink,'flat');log,curve=e.run()
    if mode=='empty':
        w.finalize(log,e.generated_signals,curve,kline=pd.DataFrame())
        assert w.manifest['visualization']['status']=='partial'
        assert all(frame.empty for _,_,frame in w._position_days)
    else:
        with pytest.raises(ValueError,match='快照'):w.finalize(log,e.generated_signals,curve,kline=e.data.kline)
        assert w.manifest['visualization']['status']=='failed'


def test_final_manifest_failure_does_not_announce_success(tmp_path,monkeypatch,capsys):
    w=writer_at(tmp_path);e=small_engine(w.snapshot);log,curve=e.run()
    original=w._write_json
    def write(name,value):
        if name=='manifest.json' and value['visualization']['status']=='complete':
            raise OSError('final manifest failed')
        original(name,value)
    monkeypatch.setattr(w,'_write_json',write)
    with pytest.raises(OSError,match='final manifest failed'):
        w.finalize(log,e.generated_signals,curve,kline=e.data.kline)
    assert not w._finished and 'HTML 报告：' not in capsys.readouterr().out
    manifest=strict_json(w.path/'manifest.json')
    assert manifest['status']=='failed' and manifest['visualization']['status']=='failed'
    assert (w.path/'quality.json').exists()
