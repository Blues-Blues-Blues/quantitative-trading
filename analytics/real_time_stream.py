"""实时/仿真结构化决策日志流（JSONL）。

StreamLogger 将 TradingStateMachine 产出的 Signal 列表（或 DataFrame）转换为
逐行标准 JSON 日志并落盘为 JSONL，供外部系统 / 复盘 / 可视化消费：

    {"symbol": "600000", "timestamp": "2024-01-03T10:00:00",
     "Final_MS": 1.52, "Global_Mod": 0.73, "Chain_Mod": 0.3,
     "Capital_Purity": 0.55, "Action": "BUY", "State": "S_push",
     "Agent_MS": ..., "ofss": ..., ...}

频率语义：信号为 Bar 级（分钟级）决策，因此日志为「每 Bar × 每标的」一行，
而非逐笔成交级别。
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# 必选字段（任务约定的标准 JSON 结构）
REQUIRED_FIELDS: List[str] = [
    "symbol", "timestamp", "Final_MS", "Global_Mod", "Chain_Mod",
    "Capital_Purity", "Action", "State",
]

# 附加字段（决策快照补充，便于归因 / 复盘复用）
EXTRA_FIELDS: List[str] = [
    "Agent_MS", "ofss", "cps", "inst_flow", "north_sync", "lock_ratio",
    "mrs", "irs", "grs",
]

# Signal.metrics 中的原始键 → 日志字段名
_METRIC_MAP = {
    "final_ms": "Final_MS",
    "global_mod": "Global_Mod",
    "chain_mod": "Chain_Mod",
    "capital_purity": "Capital_Purity",
    "agent_ms": "Agent_MS",
    "ofss": "ofss", "cps": "cps", "inst_flow": "inst_flow",
    "north_sync": "north_sync", "lock_ratio": "lock_ratio",
    "mrs": "mrs", "irs": "irs", "grs": "grs",
}

_ALL_FIELDS = REQUIRED_FIELDS + EXTRA_FIELDS


def _to_float(v: object) -> float:
    """None / NaN → 保留 NaN（json 无法序列化 NaN 时由允许标志兜底）。"""
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


class StreamLogger:
    """标准结构化决策日志流（JSONL 落盘 + 内存生成器双形态）。

    用法：
        with StreamLogger("analytics/pictures/stream.jsonl") as logger:
            for signal in signals:
                logger.log(signal)          # 写入一行并返回 JSON dict

    或仅生成不落盘：
        for row in StreamLogger.generator(signals):
            ...
    """

    def __init__(self, path: Optional[Union[str, Path]] = None,
                 allow_nan: bool = True,
                 extra_fields: Sequence[str] = EXTRA_FIELDS):
        self.path = Path(path) if path else None
        self.allow_nan = allow_nan
        self.extra_fields = list(extra_fields)
        self._fh = None
        self._count = 0

    # ---------- 上下文管理 ----------

    def __enter__(self) -> "StreamLogger":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ---------- 序列化 ----------

    @staticmethod
    def from_signal(signal: object,
                     extra_fields: Sequence[str] = EXTRA_FIELDS) -> Dict[str, object]:
        """Signal 对象 → 标准 JSON dict。缺失指标填 NaN。"""
        row = {"symbol": signal.symbol,
               "timestamp": pd.Timestamp(signal.timestamp).isoformat(),
               "Action": signal.action, "State": signal.state}
        metrics: Dict[str, object] = getattr(signal, "metrics", {}) or {}
        for metric_key, field in _METRIC_MAP.items():
            if field in extra_fields or field in REQUIRED_FIELDS:
                row[field] = _to_float(metrics.get(metric_key))
        return row

    @staticmethod
    def generator(signals: Sequence[object],
                  extra_fields: Sequence[str] = EXTRA_FIELDS) -> Iterator[Dict[str, object]]:
        """将 Signal 列表转为 JSON dict 生成器（不落盘）。"""
        for signal in signals:
            yield StreamLogger.from_signal(signal, extra_fields)

    # ---------- 写入 ----------

    def log(self, signal: object) -> Dict[str, object]:
        """写入一行 JSON 并返回该行的 dict。未指定 path 时仅返回 dict。"""
        row = self.from_signal(signal, self.extra_fields)
        if self._fh is not None:
            self._fh.write(
                json.dumps(row, ensure_ascii=False,
                           allow_nan=self.allow_nan) + "\n")
            self._count += 1
        return row

    def log_frame(self, signals: Sequence[object]) -> int:
        """批量写入，返回写入行数。"""
        n = 0
        for signal in signals:
            self.log(signal)
            n += 1
        return n

    @staticmethod
    def read(path: Union[str, Path]) -> pd.DataFrame:
        """读取 JSONL 文件 → DataFrame（用于复盘 / 测试断言）。"""
        rows = [json.loads(line) for line in
                Path(path).open("r", encoding="utf-8") if line.strip()]
        if not rows:
            return pd.DataFrame(columns=REQUIRED_FIELDS)
        return pd.DataFrame(rows)

    @property
    def count(self) -> int:
        return self._count


def to_stream_frame(signals: Sequence[object]) -> pd.DataFrame:
    """Signal 列表 → 标准日志字段 DataFrame（不落盘，便于分析）。"""
    rows = [StreamLogger.from_signal(s) for s in signals]
    if not rows:
        return pd.DataFrame(columns=_ALL_FIELDS)
    df = pd.DataFrame(rows)
    # 保证列序稳定
    cols = [c for c in _ALL_FIELDS if c in df.columns]
    return df[cols]
