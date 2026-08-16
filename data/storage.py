"""本地数据访问与存储层。

设计目标：业务代码不感知具体存储位置，统一通过「逻辑数据源 + 文件名」
读写本地数据，路径解析交由 config.data_sources 完成。

    from data import storage

    df = storage.read_frame("l2", "order_20240101.parquet")     # 读
    storage.write_frame(df, "parquet", "factors_2024.parquet")  # 写
    cfg = storage.read_yaml("industry", "industry_mapping.yaml")  # 读配置

规则说明：
- 目录缺失时读取会给出明确报错；写入会自动创建目录
- 路径尚未配置/文件不存在时，通过 DataSourceError 携带完整路径信息，
  便于后续定位并填充真实存储位置

支持格式：
    .parquet   高性能列式存储（pyarrow/fastparquet 后端，适合高频与 L2）
    .csv       轻量表格数据
    .yaml/.yml 配置与映射数据
"""

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from config.data_sources import get_data_path


class DataSourceError(RuntimeError):
    """数据访问层错误（路径缺失、文件不存在、格式不支持等）。"""


def resolve(key: str, file_name: str = "") -> Path:
    """返回逻辑数据源下某文件的完整路径（仅解析，不检查存在性）。"""
    path = get_data_path(key)
    return path / file_name if file_name else path


def exists(key: str, file_name: str = "") -> bool:
    """判断逻辑数据源（或其下某文件）是否存在。"""
    return resolve(key, file_name).exists()


def iter_files(key: str, suffix: str = "") -> List[Path]:
    """列出逻辑数据源目录下的文件（可选按后缀过滤，如 ".parquet"）。

    目录不存在时返回空列表，便于上层安全遍历（不抛错）。
    """
    d = get_data_path(key)
    if not d.is_dir():
        return []
    return [p for p in sorted(d.iterdir())
            if p.is_file() and (not suffix or p.suffix == suffix)]


def read_frame(
    key: str,
    file_name: str,
    columns: Optional[Iterable[str]] = None,
    parse_dates: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """按格式自动读取表格数据（parquet / csv）。

    :param columns: 仅读取指定列
    :param parse_dates: 仅 csv 生效；parquet 保留原生时间类型
    :raises DataSourceError: 文件不存在、格式不支持或缺少后端引擎
    """
    path = resolve(key, file_name)
    if not path.exists():
        raise DataSourceError(
            f"数据文件不存在: {path} —— 请确认数据源 '{key}' 已配置真实路径"
            f"（环境变量 QTDATA_{key.upper()}_PATH / config/data_paths.yaml），"
            f"或该文件已生成。"
        )

    cols = list(columns) if columns else None
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path, columns=cols)
        except ImportError as e:
            raise DataSourceError(
                f"读取 parquet 需要 pyarrow 或 fastparquet，请安装: pip install pyarrow"
            ) from e
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=cols, parse_dates=parse_dates)
    raise DataSourceError(
        f"不支持的表格格式: {path.suffix}（支持 .parquet / .csv）"
    )


def write_frame(df: pd.DataFrame, key: str, file_name: str) -> Path:
    """按扩展名自动写入表格数据，自动创建目录。

    :raises DataSourceError: 格式不支持或缺少后端引擎
    :return: 写入文件的完整路径
    """
    path = resolve(key, file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, engine="pyarrow")
        except ImportError as e:
            raise DataSourceError(
                f"写入 parquet 需要 pyarrow 或 fastparquet，请安装: pip install pyarrow"
            ) from e
    elif path.suffix == ".csv":
        df.to_csv(path, encoding="utf-8", index=True)
    else:
        raise DataSourceError(
            f"不支持的表格格式: {path.suffix}（支持 .parquet / .csv）"
        )
    return path


def read_yaml(key: str, file_name: str) -> dict:
    """读取 YAML 配置/映射数据。

    :raises DataSourceError: 文件不存在或缺少 PyYAML
    """
    path = resolve(key, file_name)
    if not path.exists():
        raise DataSourceError(f"配置文件不存在: {path}")
    try:
        import yaml
    except ImportError as e:
        raise DataSourceError("缺少 PyYAML，请先安装: pip install pyyaml") from e
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
