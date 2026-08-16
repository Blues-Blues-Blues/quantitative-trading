"""本地数据源注册与路径解析。

背景：项目后续将大量依赖本地存储数据，但具体存储位置在开发阶段尚未确定。
本模块提供「逻辑数据源名 → 实际路径」的统一解析，业务代码只依赖逻辑名；
待位置确定后，通过环境变量 / 配置文件 / 运行时注册三种方式注入真实路径，
业务代码无需改动。

路径解析优先级（高 → 低）：
    1. 环境变量        QTDATA_<KEY>_PATH（如 QTDATA_L2_PATH，部署级覆盖）
    2. 配置文件        config/data_paths.yaml（用户配置级覆盖，可选项）
    3. 运行时注册      set_data_path(key, path)（代码内临时切换/后续集成）
    4. 默认路径        项目根目录下的默认相对路径（兜底）

内置逻辑数据源（KEY）及默认位置：
    daily_cache   日线行情缓存              data/mock_data
    pictures      图片输出目录              analytics/pictures
    l2            Level-2 快照/逐笔数据     data/l2
    macro         宏观/外盘/商品/汇率数据    data/macro
    industry      产业链映射                config
    parquet       通用高性能 Parquet 存储   data/parquet
    strategy      策略参数与中间结果        data/strategy
    logs          结构化实时决策日志        data/logs
"""

import os
from pathlib import Path
from typing import Dict, Optional

# 项目根目录（本文件位于 config/ 下，取上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 环境变量前缀：QTDATA_<KEY>_PATH
_ENV_PREFIX = "QTDATA_"

# 路径覆盖配置文件（可选，见同目录 data_paths.yaml）
_DATA_PATHS_CONFIG = Path(__file__).resolve().parent / "data_paths.yaml"

# 逻辑数据源 → 默认相对路径（相对项目根目录）
_DEFAULT_PATHS: Dict[str, str] = {
    "daily_cache": "data/mock_data",
    "pictures": "analytics/pictures",
    "l2": "data/l2",
    "macro": "data/macro",
    "industry": "config",
    "parquet": "data/parquet",
    "strategy": "data/strategy",
    "logs": "data/logs",
}

# 运行时注册的路径覆盖（优先级高于默认路径）
_registry: Dict[str, Path] = {}


def register_source(key: str, default_rel_path: str) -> None:
    """注册新的逻辑数据源及默认相对路径（供扩展自定义数据源）。"""
    if not key or not isinstance(key, str):
        raise ValueError(f"数据源 key 必须为非空字符串，当前: {key!r}")
    _DEFAULT_PATHS[key] = default_rel_path


def set_data_path(key: str, path: str) -> None:
    """运行时注册实际路径（后续集成/测试时调用，可反复覆盖）。

    优先级低于环境变量与 data_paths.yaml 配置，高于默认路径。
    """
    if key not in _DEFAULT_PATHS:
        raise KeyError(f"未知数据源 key: {key}，可用: {sorted(_DEFAULT_PATHS)}")
    _registry[key] = Path(path)


def _env_path(key: str) -> Optional[Path]:
    """读取环境变量 QTDATA_<KEY>_PATH。"""
    env_name = f"{_ENV_PREFIX}{key.upper()}_PATH"
    val = os.environ.get(env_name)
    return Path(val) if val else None


def _yaml_paths() -> Dict[str, Path]:
    """读取 config/data_paths.yaml 中的路径覆盖。

    文件不存在或未安装 PyYAML 时返回空 dict（该配置为可选项，静默跳过）。
    """
    if not _DATA_PATHS_CONFIG.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(_DATA_PATHS_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {k: Path(v) for k, v in raw.items() if isinstance(v, str)}


def get_data_path(key: str) -> Path:
    """解析逻辑数据源的完整路径（返回 Path，目录可能尚不存在）。

    :raises KeyError: key 未注册
    """
    if key not in _DEFAULT_PATHS:
        raise KeyError(f"未知数据源 key: {key}，可用: {sorted(_DEFAULT_PATHS)}")

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p

    # 1) 环境变量（最高优先级）
    p = _env_path(key)
    if p is not None:
        return _abs(p)

    # 2) 配置文件 data_paths.yaml
    p = _yaml_paths().get(key)
    if p is not None:
        return _abs(p)

    # 3) 运行时注册
    p = _registry.get(key)
    if p is not None:
        return _abs(p)

    # 4) 默认相对路径
    return _abs(Path(_DEFAULT_PATHS[key]))
