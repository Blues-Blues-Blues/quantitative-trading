"""统一特征计算与归一化调度器（FeatureEngine）。

输入 DataSlice，输出覆盖全部特征列的个股长表 DataFrame
（DatetimeIndex + symbol 列），列集合见 FEATURE_COLS。

调度流程：
1. TimeAligner 对齐：macro T-1、north_margin/两融 T-1、龙虎榜 T+1（avail_date）
2. 分钟级因子（当日可用）：资金流（retail/inst/youzi）、OFSS 成分与综合、
   PSS、MRS、GRS、IRS
3. 日频因子（T-1 对齐后 ffill 到分钟轴）：North_Sync、Margin_Pressure、
   CPS 成分与综合
4. 龙虎榜 T+1 因子（dt_net），并用 verify_no_lookahead 断言
5. 组装长表：任何数据表缺失时，对应特征列置 NaN 并告警，绝不中断

所有权重与窗口参数由 AgentProfiling / MicroStructure / Environment 构造
入参注入，便于 Optuna 超参数寻优。
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from data.aligner import TimeAligner
from data.dataslice import SYMBOL, TRADE_DATE, DataSlice
from indicators.agent_profiling import AgentProfiling
from indicators.environment import Environment
from indicators.microstructure import MicroStructure

logger = logging.getLogger("indicators.feature_engine")

# ---- 特征持久化缓存（整区间 + 签名 key）----
# 命中条件 = 参数签名 + 数据指纹 + 区间 + 标的，全部一致才复用，
# 跳过高耗时的原始逐笔（tick/l2_snapshot）加载与因子重算。
_CACHE_SCHEMA_VERSION = "3"   # FEATURE_COLS / 因子计算逻辑变化时递增（强制全部失效）
                              # 1→2：特征数值列统一强制 float64（修复 object dtype 往返不一致）
                              # 2→3：增量拼接去重改为按行键 (ts, symbol)（索引仅 ts，长表多标）
                              #      （原按 index 去重会删掉同 ts 的第二个标的 → 数据减半）
_ALIGN_VERSION = "1"          # TimeAligner 行为变化时递增
_DEFAULT_CACHE_DIR = (Path(__file__).resolve().parent.parent
                      / "data" / "feature_cache")
# 数据指纹扫描目录（用户补充数据后指纹变化 → 旧缓存自动失效）
_DEFAULT_FINGERPRINT_DIRS = (
    Path(__file__).resolve().parent.parent / "data" / "data1" / "data",
    Path(__file__).resolve().parent.parent / "data" / "data2",
)
_fingerprint_memo: Dict[str, str] = {}

FEATURE_COLS: List[str] = [
    SYMBOL,
    # 资金主体分层
    "retail_flow", "inst_flow", "youzi_flow",
    "north_sync", "margin_pressure",
    # 订单流与筹码微观结构
    "obi", "ar", "cancel_ratio", "big_flow", "ofss",
    "lock_ratio", "accum_delta", "panic_ratio", "drift", "cps",
    "pss",
    # 宏观与行业环境共振
    "mrs", "grs", "irs", "global_mod", "chain_mod",
    # 龙虎榜（T+1 可用）
    "dt_net",
]

# 日频因子的值列（chip / 北向 / 两融）
_CHIP_COLS = ["lock_ratio", "accum_delta", "panic_ratio", "drift"]

# 数值特征列（symbol 除外）：缓存往返前统一强制 float64
_NUMERIC_COLS = [c for c in FEATURE_COLS if c != SYMBOL]


class FeatureEngine:
    """特征计算调度器。

    :param agent:   资金主体分层计算器（缺省 AgentProfiling()）
    :param micro:   微观结构计算器（缺省 MicroStructure()）
    :param env:     环境共振计算器（缺省 Environment()）
    :param aligner: 时间对齐器（缺省 TimeAligner()）
    :param symbol_to_industry: {个股代码: 行业名}，用于把 IRS 映射到个股；
        不提供时 irs / chain_mod 置 NaN
    """

    def __init__(
        self,
        agent: Optional[AgentProfiling] = None,
        micro: Optional[MicroStructure] = None,
        env: Optional[Environment] = None,
        aligner: Optional[TimeAligner] = None,
        symbol_to_industry: Optional[Dict[str, str]] = None,
    ) -> None:
        self.agent = agent or AgentProfiling()
        self.micro = micro or MicroStructure()
        self.env = env or Environment()
        self.aligner = aligner or TimeAligner()
        self.symbol_to_industry = symbol_to_industry or {}

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def compute(self, ds: DataSlice) -> pd.DataFrame:
        """计算全部特征并组装为个股长表。

        :return: DatetimeIndex + 列 = FEATURE_COLS
        """
        aligned = self.aligner.align_slice(ds)
        axis = aligned.time_axis()

        # 基准长表（分钟轴 × 标的）
        base = aligned.kline.reset_index()
        base.columns = ["ts"] + list(base.columns[1:])
        feat = base[["ts", SYMBOL]].copy()

        # ---- 分钟级因子 ----
        feat = self._minute_factors(feat, aligned)

        # ---- 日频因子（T-1 对齐 → ffill 分钟）----
        feat = self._daily_factors(feat, aligned, axis)

        # ---- 环境共振 ----
        feat = self._environment_factors(feat, aligned)

        # ---- 龙虎榜 T+1 因子 + 防未来断言 ----
        feat = self._dragon_tiger(feat, aligned, axis)

        # 固定列顺序输出；数值列统一 float64——上游 object dtype
        #（如快照量列）会破坏 parquet 缓存往返一致性，且拖慢下游计算
        feat = feat.set_index("ts").reindex(columns=FEATURE_COLS)
        feat[_NUMERIC_COLS] = feat[_NUMERIC_COLS].apply(
            pd.to_numeric, errors="coerce")
        return feat

    # ------------------------------------------------------------------
    # 分钟级因子
    # ------------------------------------------------------------------

    def _minute_factors(self, feat: pd.DataFrame, ds: DataSlice) -> pd.DataFrame:
        # 资金流（含归一化基准）
        if ds.tick_trades is not None and not ds.tick_trades.empty:
            norm_base = self._norm_base(ds.kline, ds.time_axis())
            flows = self.agent.net_flows(ds.tick_trades, norm_base)
            feat = feat.merge(flows.reset_index().rename(columns={"index": "ts"}),
                              on=["ts", SYMBOL], how="left")
        else:
            logger.warning("缺少 tick_trades，retail/inst/youzi_flow 置 NaN")

        # OFSS 成分与综合
        if (ds.l2_snapshot is not None and not ds.l2_snapshot.empty) \
                or (ds.tick_trades is not None and not ds.tick_trades.empty):
            comp = self.micro.ofss_components(ds)
            comp["ofss"] = self.micro.ofss(comp)
            comp = comp.reset_index().rename(columns={"index": "ts"})
            feat = feat.merge(comp[["ts", SYMBOL, "obi", "ar", "cancel_ratio",
                                    "big_flow", "ofss"]],
                              on=["ts", SYMBOL], how="left")
        else:
            logger.warning("缺少 l2_snapshot 且缺少 tick_trades，OFSS 相关列置 NaN")

        # PSS（与 kline 同 index 的 Series；长表按 [ts, symbol] 对齐补 symbol 列）
        pss = self.micro.pss(ds.kline).rename("pss").reset_index()
        pss = pss.rename(columns={"index": "ts"})
        pss[SYMBOL] = ds.kline[SYMBOL].to_numpy()  # pss 与 kline 行序一致
        feat = feat.merge(pss, on=["ts", SYMBOL], how="left")
        return feat

    # ------------------------------------------------------------------
    # 日频因子（T-1 对齐后 ffill 到分钟轴）
    # ------------------------------------------------------------------

    def _daily_factors(self, feat: pd.DataFrame, ds: DataSlice,
                       axis: pd.DatetimeIndex) -> pd.DataFrame:
        # 北向 / 两融
        if ds.north_margin is not None and not ds.north_margin.empty:
            ns = self.agent.north_sync(ds.north_margin)
            mp = self.agent.margin_pressure(ds.north_margin)
            daily = ns.merge(mp, on=[TRADE_DATE, SYMBOL], how="outer")
            aligned_d = self._ffill_daily(daily, axis, ["north_sync", "margin_pressure"])
            feat = feat.merge(aligned_d, on=["ts", SYMBOL], how="left")
        else:
            logger.warning("缺少 north_margin，north_sync / margin_pressure 置 NaN")

        # CPS 筹码分量
        if ds.tick_trades is not None and not ds.tick_trades.empty:
            chip = self.micro.chip_components(ds)
            aligned_c = self._ffill_daily(chip, axis, _CHIP_COLS)
            aligned_c["cps"] = self.micro.cps(aligned_c)
            feat = feat.merge(aligned_c[["ts", SYMBOL] + _CHIP_COLS + ["cps"]],
                              on=["ts", SYMBOL], how="left")
        else:
            logger.warning("缺少 tick_trades，CPS 相关列置 NaN")
        return feat

    def _ffill_daily(self, daily: pd.DataFrame, axis: pd.DatetimeIndex,
                     value_cols: List[str]) -> pd.DataFrame:
        """日频长表按 T-1 可用时点对齐并 ffill 到分钟轴。"""
        rows = []
        for sym, g in daily.groupby(SYMBOL):
            a = self.aligner.align_external(g, axis, value_cols,
                                            date_col=TRADE_DATE)
            a[SYMBOL] = sym
            rows.append(a)
        out = pd.concat(rows)
        return out.reset_index().rename(columns={"index": "ts"})

    def _norm_base(self, kline: pd.DataFrame, axis: pd.DatetimeIndex) -> pd.DataFrame:
        """资金流归一化基准（日频 → T-1 对齐 → 分钟轴长表 [ts, symbol, norm_base]）。"""
        k = kline.copy()
        k["day"] = k.index.normalize()
        if self.agent.norm_by == "amount":
            daily = k.groupby(["day", SYMBOL])["amount"].sum().reset_index()
            daily = daily.rename(columns={"day": TRADE_DATE, "amount": "norm_base"})
            daily["norm_base"] = (
                daily.groupby(SYMBOL)["norm_base"]
                .transform(lambda s: s.rolling(self.agent.norm_days,
                                               min_periods=1).mean()))
        else:  # mcap：流通市值按日末值
            daily = k.groupby(["day", SYMBOL])["float_market_cap"].last().reset_index()
            daily = daily.rename(columns={"day": TRADE_DATE,
                                          "float_market_cap": "norm_base"})
        return self._ffill_daily(daily, axis, ["norm_base"])

    # ------------------------------------------------------------------
    # 环境共振
    # ------------------------------------------------------------------

    def _environment_factors(self, feat: pd.DataFrame, ds: DataSlice) -> pd.DataFrame:
        # MRS（取首个指数代码作为全市场状态）
        mrs = self.env.mrs(ds.index_min, ds.breadth)
        if not mrs.empty:
            code = mrs["index_code"].iloc[0]
            mrs = mrs[mrs["index_code"] == code][["mrs"]].reset_index()
            mrs = mrs.rename(columns={"index": "ts"})
            feat = feat.merge(mrs, on="ts", how="left")
        else:
            logger.warning("MRS 为空，置 NaN")

        # GRS / Global_Mod（macro 已 T-1 对齐）
        grs = self.env.grs(ds.macro)
        if not grs.empty:
            grs["global_mod"] = self.env.global_mod(grs)
            grs = grs.reset_index().rename(columns={"index": "ts"})
            feat = feat.merge(grs, on="ts", how="left")
        else:
            logger.warning("GRS 为空，grs / global_mod 置 NaN")

        # IRS / Chain_Mod（需 symbol→industry 映射）
        if self.symbol_to_industry:
            irs = self.env.irs(ds.industry, ds.macro)
            if not irs.empty:
                irs["chain_mod"] = self.env.chain_mod(irs)
                irs = irs.reset_index().rename(columns={"index": "ts"})
                map_df = pd.DataFrame({
                    SYMBOL: list(self.symbol_to_industry.keys()),
                    "industry": list(self.symbol_to_industry.values()),
                })
                per_symbol = irs.merge(map_df, on="industry", how="inner")
                feat = feat.merge(per_symbol.drop(columns="industry"),
                                  on=["ts", SYMBOL], how="left")
                return feat
            logger.warning("IRS 为空，irs / chain_mod 置 NaN")
        else:
            logger.warning("未提供 symbol_to_industry，irs / chain_mod 置 NaN")
        feat["irs"] = pd.NA
        feat["chain_mod"] = pd.NA
        return feat

    # ------------------------------------------------------------------
    # 龙虎榜 T+1 因子
    # ------------------------------------------------------------------

    def _dragon_tiger(self, feat: pd.DataFrame, ds: DataSlice,
                      axis: pd.DatetimeIndex) -> pd.DataFrame:
        if ds.dragon_tiger is None or ds.dragon_tiger.empty:
            feat["dt_net"] = pd.NA
            return feat

        d = ds.dragon_tiger.dropna(subset=["avail_date"]).copy()
        if d.empty:
            feat["dt_net"] = pd.NA
            return feat
        agg = d.groupby([SYMBOL, "avail_date"])["net_amount"].sum().reset_index()

        # avail_date（披露次日 00:00）起可用：按分钟轴日期 reindex + ffill
        rows = []
        for sym, g in agg.groupby(SYMBOL):
            s = g.set_index("avail_date")["net_amount"]
            mapped = s.reindex(axis.normalize(), method="ffill")
            rows.append(pd.DataFrame(
                {"ts": axis, SYMBOL: sym,
                 "dt_net": mapped.to_numpy(),
                 "dt_avail": mapped.index.to_numpy()}))
        out = pd.concat(rows)

        # 防未来函数断言：每行使用时间(ts)必须晚于其数据可用日
        TimeAligner.verify_no_lookahead(out["ts"], out["dt_avail"],
                                        name="dragon_tiger/dt_net")
        feat = feat.merge(out[["ts", SYMBOL, "dt_net"]], on=["ts", SYMBOL],
                          how="left")
        return feat

    # ------------------------------------------------------------------
    # 持久化缓存（整区间 + 签名 key，命中跳过逐笔加载与因子重算）
    # ------------------------------------------------------------------

    @staticmethod
    def _serializable(obj: object) -> object:
        """只保留可序列化的基本类型（过滤 logger 等运行时对象）。"""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple, set)):
            return [FeatureEngine._serializable(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): FeatureEngine._serializable(v)
                    for k, v in obj.items()}
        return str(obj)

    def _params_signature(self) -> str:
        """参数签名：agent / micro / env 构造参数 + 行业映射，排序序列化哈希。

        任一窗口 / 阈值参数变化都会改变签名 → 缓存 key 变化 → 自动重算。
        """
        payload = {
            "agent": self._serializable(vars(self.agent)),
            "micro": self._serializable(vars(self.micro)),
            "env": self._serializable(vars(self.env)),
            "symbol_to_industry": self._serializable(dict(self.symbol_to_industry)),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]

    @classmethod
    def data_fingerprint(cls, dirs: Sequence[Path]) -> str:
        """数据目录指纹：max(mtime_ns) + 总字节数 + 文件数。

        用户补充/修改数据后 mtime 或大小变化 → 指纹变化 → 旧缓存失效。
        同进程内结果记忆化，避免每次回测重复扫描 3 万文件。
        """
        dirs = tuple(str(d) for d in dirs)
        if dirs in _fingerprint_memo:
            return _fingerprint_memo[dirs]
        max_mt, total, n = 0, 0, 0
        for d in dirs:
            root = Path(d)
            if not root.is_dir():
                continue
            for base, _, files in os.walk(root):
                for f in files:
                    try:
                        st = (Path(base) / f).stat()
                    except OSError:
                        continue
                    max_mt = max(max_mt, st.st_mtime_ns)
                    total += st.st_size
                    n += 1
        fp = f"{max_mt:x}_{total:x}_{n}"
        _fingerprint_memo[dirs] = fp
        return fp

    def cache_file_name(self, start: str, end: str,
                        symbols: Sequence[str], fp: str) -> str:
        """结构化缓存文件名（可解析出 start/end，支持增量定位）。

        feat.<schema>.<align>.<params16>.<start>.<end>.<sym8>.<fp16>.parquet
        """
        p16 = self._params_signature()
        sym = hashlib.sha256(
            ",".join(sorted(symbols)).encode()).hexdigest()[:8]
        return (f"feat.{_CACHE_SCHEMA_VERSION}.{_ALIGN_VERSION}.{p16}."
                f"{start}.{end}.{sym}.{fp}.parquet")

    def cache_path(self, ds: DataSlice, fingerprint_dirs=None) -> Optional[Path]:
        """计算当前 DataSlice 应命中的缓存路径（不含存在性判断）。"""
        meta = ds.meta
        start, end = str(meta.get("start", "")), str(meta.get("end", ""))
        symbols = list(meta.get("symbols", []))
        if not start or not end or not symbols:
            return None
        fp = self.data_fingerprint(fingerprint_dirs or _DEFAULT_FINGERPRINT_DIRS)
        return (_DEFAULT_CACHE_DIR / self.cache_file_name(start, end, symbols, fp))

    def cache_exists(self, start: str, end: str, symbols: Sequence[str],
                     fingerprint_dirs=None) -> bool:
        """预检：该区间+参数+数据指纹下是否存在精确命中缓存（供加载前跳过 tick）。"""
        fp = self.data_fingerprint(fingerprint_dirs or _DEFAULT_FINGERPRINT_DIRS)
        p = _DEFAULT_CACHE_DIR / self.cache_file_name(start, end, symbols, fp)
        return p.exists()

    @staticmethod
    def _read_features(path: Path) -> pd.DataFrame:
        feat = pd.read_parquet(path)
        if "ts" in feat.columns:
            feat = feat.set_index("ts")
        # pandas 2.x 读回可能降为 datetime64[us]，与分钟轴（ns）强制对齐
        feat.index = pd.DatetimeIndex(feat.index).as_unit("ns")
        return feat.reindex(columns=FEATURE_COLS)

    @staticmethod
    def _write_features(feat: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        feat.reset_index().rename(columns={"index": "ts"}).to_parquet(
            path, index=False)

    def _incremental_candidate(self, start: str, end: str,
                               symbols: Sequence[str], fp: str) -> Optional[Path]:
        """增量候选：同 参数/对齐/起点/标的/指纹、end 严格更小的旧缓存。

        区间向后延长（同起点、end 变大）时复用旧段，只补算新段。
        """
        p16 = self._params_signature()
        sym = hashlib.sha256(
            ",".join(sorted(symbols)).encode()).hexdigest()[:8]
        prefix = f"feat.{_CACHE_SCHEMA_VERSION}.{_ALIGN_VERSION}.{p16}.{start}."
        best: Optional[Path] = None
        best_end = ""
        if _DEFAULT_CACHE_DIR.is_dir():
            for f in _DEFAULT_CACHE_DIR.glob(f"{prefix}*.parquet"):
                parts = f.name.split(".")
                # feat, V, A, p16, start, end, sym, fp, parquet
                if len(parts) != 9:
                    continue
                _, _v, _a, _p, _s, e, s8, f8, _ext = parts
                if e >= end or s8 != sym or f8 != fp:
                    continue
                if e > best_end:
                    best, best_end = f, e
        return best

    def compute_cached(self, ds: DataSlice,
                       fingerprint_dirs: Optional[Sequence[Path]] = None,
                       cache_dir: Optional[Path] = None) -> pd.DataFrame:
        """compute() 的持久化缓存版本。

        1. 冒烟数据（meta.smoke）不走缓存（秒级，无需落盘）
        2. 精确命中（参数+指纹+区间+标的全一致）→ 直接读取，跳过逐笔计算
        3. 增量命中（同起点向后延长）→ 复用旧段 + 补算新段，替换旧缓存
        4. 未命中 → 全量计算并落盘

        增量语义说明：日频因子（T-1 ffill）取值依赖整个对齐轴，且分钟
        因子滚动窗口需要历史前缀，因此新段仍按全区间计算后截取——增量
        省去的是旧段被重算导致的边界不一致与整表 I/O；计算量仍随总区间
        增长（真正按日增量需因子层分区改造，后续可做）。
        """
        if ds.meta.get("smoke"):
            return self.compute(ds)
        meta = ds.meta
        start, end = str(meta.get("start", "")), str(meta.get("end", ""))
        symbols = sorted(meta.get("symbols", []))
        if not start or not end or not symbols:
            logger.warning("ds.meta 缺少 start/end/symbols，跳过特征缓存走全量计算")
            return self.compute(ds)
        fp = self.data_fingerprint(fingerprint_dirs or _DEFAULT_FINGERPRINT_DIRS)
        cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        path = cache_dir / self.cache_file_name(start, end, symbols, fp)

        if path.exists():
            logger.info("特征缓存命中：%s（直接读取，跳过逐笔计算）", path.name)
            return self._read_features(path)

        incr = self._incremental_candidate(start, end, symbols, fp)
        if incr is not None:
            logger.info("特征缓存增量扩展：%s → %s（复用旧段，补算新段）",
                        incr.name, path.name)
            old = self._read_features(incr)
            new = self.compute(ds)
            merged = pd.concat([old, new[new.index > old.index.max()]])
            # 长表行键是 (ts, symbol) 而非 ts：索引仅 ts 的多标表不能用
            # index 去重（会删掉同 ts 的第二个标的）。按构造旧段与新段
            # 时间不重叠，这里按行键去重仅为边界安全兜底
            merged = merged[
                ~merged.reset_index().duplicated(
                    subset=["ts", SYMBOL], keep="first").to_numpy()]
            merged = merged.sort_index()
            self._write_features(merged, path)
            incr.unlink(missing_ok=True)  # 新缓存取代旧缓存
            return merged

        logger.info("特征缓存未命中，全量计算并落盘：%s", path.name)
        feat = self.compute(ds)
        self._write_features(feat, path)
        return feat
