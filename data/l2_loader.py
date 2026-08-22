"""Level-2 数据解析器：data1 万得 L2 parquet → 标准分钟级表。

数据布局（data/data1/data/{市场}/{代码}.{市场}/{日期}.{类型}.parquet）：
    - 类型：行情（10 档快照，66 列）/ 逐笔成交 / 逐笔委托
    - 时间编码：自然日 = YYYYMMDD（int）；时间 = HHMMSSmmm（int，前导 0 省略，
      92500780 → 09:25:00.780）
    - 价格：整数 ÷10000 得元（63800 → 6.38）；成交量：股；成交额：元

职责：
1. 行情快照 → 分钟 K 线（仅 A 股交易时段 09:30~11:30 + 13:00~15:00；
   集合竞价段 09:15~09:25 的参考价快照不进入 K 线）
2. 行情快照 → 标准盘口快照（默认前 5 档，depth=10 时输出 10 档；
   同一分钟多帧取最后一帧）
3. 逐笔成交 → 标准逐笔表：
   - 时间戳单调性校验（乱序 → 告警 + 自动重排，不中断）
   - 异常单过滤（价格/数量 ≤ 0、BS 标志非法 → 丢弃并告警）
   - 09:25 开盘集合竞价撮合成交标记 is_open_auction=True（保留为开盘事件）
4. 逐笔委托 → 标准委托表（含 09:15 起的集合竞价委托；A=委托 / D=撤单）

输出均为严格类型：index=DatetimeIndex（分钟轴），symbol=str6，
价格/数量=float64，方向=int8，布尔列=bool。schema 见各方法的 docstring。
"""

import logging
from datetime import time as dtime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from data.dataslice import SYMBOL

logger = logging.getLogger("data.l2_loader")

# 万得价格单位：整数 ÷10000 得元
PRICE_SCALE = 10000.0

# 开盘集合竞价撮合时段（09:25:00~09:25:59 的成交为开盘撮合结果）
_OPEN_AUCTION_MIN = pd.Timestamp("09:25:00").time()
_OPEN_AUCTION_END = pd.Timestamp("09:26:00").time()

# 行情快照的 10 档盘口原始列名（前 10 档，depth≤10 时按需切片）
_SNAP_PRICE_COLS = [f"申买价{i}" for i in range(1, 11)] + \
                   [f"申卖价{i}" for i in range(1, 11)]
_SNAP_VOL_COLS = [f"申买量{i}" for i in range(1, 11)] + \
                 [f"申卖量{i}" for i in range(1, 11)]


def _hmss_to_timedelta(hmss: pd.Series) -> pd.Timedelta:
    """HHMMSSmmm int → Timedelta（补零到 9 位：92500780 → 09:25:00.780）。"""
    s = hmss.astype(str).str.zfill(9)
    hh = s.str[:2].astype(int)
    mm = s.str[2:4].astype(int)
    ss = s.str[4:6].astype(int)
    ms = s.str[6:9].astype(int)
    return (pd.to_timedelta(hh * 3600 + mm * 60 + ss, unit="s")
            + pd.to_timedelta(ms, unit="ms"))


def _quote_paths(data1_root: Path, symbol6: str, kind: str,
                 start: str, end: str) -> List[Path]:
    """列出某标的某类型文件（按文件名日期过滤 [start, end]，含两端）。"""
    for mkt in ("SH", "SZ", "BJ"):
        d = data1_root / mkt / f"{symbol6}.{mkt}"
        if d.is_dir():
            break
    else:
        return []
    s, e = start.replace("-", ""), end.replace("-", "")
    out = []
    for f in sorted(d.glob(f"*.{kind}.parquet")):
        day = f.name.split(".")[0]
        if s <= day <= e:
            out.append(f)
    return out


class L2DataLoader:
    """data1 万得 Level-2 parquet → 标准分钟级长表。

    :param data1_root: data/data1/data 目录
    """

    def __init__(self, data1_root: Path) -> None:
        self.root = Path(data1_root)

    # ------------------------------------------------------------------
    # 分钟 K 线
    # ------------------------------------------------------------------

    def load_kline(self, symbols: List[str], start: str, end: str,
                   freq: str = "1min") -> pd.DataFrame:
        """行情快照 → 分钟 K 线（长表，列见 KLINE 基础列 + symbol）。

        - 每笔成交触发的快照行（close>0）重采样为 freq 周期 OHLCV
        - 只保留 A 股交易时段（09:30~11:30 + 13:00~15:00）
        - 集合竞价段（09:14~09:25）参考价快照不参与 K 线

        输出列：[symbol, open, high, low, close, volume, amount]
        """
        frames: List[pd.DataFrame] = []
        for sym in symbols:
            for f in _quote_paths(self.root, sym, "行情", start, end):
                df = self._read_parquet(f, "行情")
                if df is None or df.empty:
                    continue
                df = self._clean_raw(df, f.name)
                ts = self._file_timestamp(df)
                try:
                    p = (pd.to_numeric(df["成交价"], errors="coerce")
                         .to_numpy() / PRICE_SCALE)
                    s = pd.DataFrame({
                        "open": p, "high": p, "low": p, "close": p,
                        "volume": pd.to_numeric(df["成交量"], errors="coerce").to_numpy(),
                        "amount": pd.to_numeric(df["成交额"], errors="coerce").to_numpy(),
                    }, index=ts)
                except KeyError as e:
                    logger.warning("行情文件 %s 缺列 %s，跳过", f.name, e)
                    continue
                s["amount"] = s["amount"].clip(lower=0.0)  # 成交额不应为负（脏数据防抖）
                s = s[s["close"] > 0]  # 无成交的快照行不参与
                if s.empty:
                    continue
                bar = s.resample(freq).agg(
                    open=("open", "first"), high=("high", "max"),
                    low=("low", "min"), close=("close", "last"),
                    volume=("volume", "sum"), amount=("amount", "sum"))
                tm = bar.index.time
                mask = ((tm >= dtime(9, 30)) & (tm <= dtime(11, 30))) | \
                       ((tm >= dtime(13, 0)) & (tm <= dtime(15, 0)))
                bar = bar[mask]
                bar[SYMBOL] = sym
                frames.append(bar)
        if not frames:
            logger.warning("data1 无 %s 在 %s~%s 的行情数据，返回空表",
                           symbols, start, end)
            return pd.DataFrame(columns=[
                "open", "high", "low", "close", "volume", "amount", SYMBOL])
        out = pd.concat(frames).sort_index()
        return out[["open", "high", "low", "close", "volume", "amount", SYMBOL]]

    # ------------------------------------------------------------------
    # 盘口快照
    # ------------------------------------------------------------------

    def load_l2_snapshot(self, symbols: List[str], start: str, end: str,
                         depth: int = 5, freq: str = "1min") -> pd.DataFrame:
        """行情快照 10 档盘口 → 标准快照表（同一分钟取最后一帧）。

        :param depth: 档位深度，5 或 10（默认 5，列名 bid1_p...bid5_v / ask1_p...ask5_v）
        :return: 长表 [symbol, bid1_p..bidN_v, ask1_p..askN_v]
        """
        if depth not in (5, 10):
            raise ValueError(f"depth 仅支持 5 或 10，收到 {depth}")
        frames: List[pd.DataFrame] = []
        for sym in symbols:
            for f in _quote_paths(self.root, sym, "行情", start, end):
                df = self._read_parquet(f, "行情")
                if df is None or df.empty:
                    continue
                df = self._clean_raw(df, f.name)
                ts = self._file_timestamp(df)
                try:
                    n = depth
                    # 列次序：前 10 个为买价/买量，后 10 个为卖价/卖量
                    pc = _SNAP_PRICE_COLS[:n] + _SNAP_PRICE_COLS[10:10 + n]
                    vc = _SNAP_VOL_COLS[:n] + _SNAP_VOL_COLS[10:10 + n]
                    p = pd.DataFrame(
                        df[pc].to_numpy() / PRICE_SCALE,
                        index=ts.floor("min"), columns=pc)
                    v = pd.DataFrame(
                        df[vc].to_numpy(),
                        index=ts.floor("min"), columns=vc)
                except KeyError as e:
                    logger.warning("行情文件 %s 缺列 %s，跳过", f.name, e)
                    continue
                s = pd.concat([p, v], axis=1)
                s = s.groupby(level=0).last()  # 同一分钟多帧取最后一帧
                if freq != "1min":
                    s = s.resample(freq).last()
                s[SYMBOL] = sym
                frames.append(s)
        if not frames:
            return pd.DataFrame()
        snap = pd.concat(frames).sort_index()
        out = pd.DataFrame(index=snap.index)
        for i in range(1, depth + 1):
            out[f"bid{i}_p"] = snap[f"申买价{i}"]
            out[f"bid{i}_v"] = snap[f"申买量{i}"]
            out[f"ask{i}_p"] = snap[f"申卖价{i}"]
            out[f"ask{i}_v"] = snap[f"申卖量{i}"]
        out[SYMBOL] = snap[SYMBOL]
        return out

    # ------------------------------------------------------------------
    # 逐笔成交
    # ------------------------------------------------------------------

    def load_tick_trades(self, symbols: List[str], start: str,
                         end: str) -> pd.DataFrame:
        """逐笔成交 → 标准逐笔表。

        输出列：[symbol, price, volume, turnover, side, is_cancel, is_open_auction]
        - price：元（÷10000）；volume：股；turnover：元 = price × volume
        - side：1=主动买 / -1=主动卖 / 0=中性（BS 标志非法时丢弃）
        - is_cancel：False（成交记录非撤单）
        - is_open_auction：09:25 开盘集合竞价撮合成交标记 True
        - 时间戳：毫秒级原始时间校验单调性（乱序 → 告警+重排），
          随后 floor 到分钟与 K 线轴对齐
        - 异常单（价格/数量 ≤ 0、BS 标志非法）过滤并告警，不中断
        """
        frames: List[pd.DataFrame] = []
        n_orders_bad = n_ts_reorder = n_ts_total = 0
        for sym in symbols:
            for f in _quote_paths(self.root, sym, "逐笔成交", start, end):
                df = self._read_parquet(f, "逐笔成交")
                if df is None or df.empty:
                    continue
                df = self._clean_raw(df, f.name)
                try:
                    price_raw = pd.to_numeric(df["成交价格"], errors="coerce")
                    vol = pd.to_numeric(df["成交数量"], errors="coerce")
                    bs = df["BS标志"].astype(str)
                    ts = self._file_timestamp(df)
                except KeyError as e:
                    logger.warning("逐笔文件 %s 缺列 %s，跳过", f.name, e)
                    continue

                # ---- 时间戳单调性校验（毫秒级原始序列）----
                n_ts_total += len(ts)
                if not ts.is_monotonic_increasing:
                    # pandas 无向量化乱序计数，用排序对比
                    k = int((ts.to_numpy() != np.sort(ts.to_numpy())).sum())
                    n_ts_reorder += k
                    if k:
                        logger.warning(
                            "逐笔成交 %s 时间戳乱序 %d/%d 行（%.2f%%），自动重排",
                            f.name, k, len(ts), 100.0 * k / len(ts))
                    ts = pd.DatetimeIndex(np.sort(ts.to_numpy()))

                price = price_raw.to_numpy() / PRICE_SCALE
                vol = vol.to_numpy()
                side = np.select([bs == "B", bs == "S"], [1, -1], default=0)

                # ---- 异常单过滤：价格/数量 ≤ 0、BS 标志非法 ----
                bad = (price <= 0) | (vol <= 0) | (side == 0) | \
                      np.isnan(price) | np.isnan(vol)
                bad = np.asarray(bad, dtype=bool)
                nb = int(bad.sum())
                n_orders_bad += nb
                if nb:
                    price, vol, side = price[~bad], vol[~bad], side[~bad]
                    ts = ts[~bad]

                if len(ts) == 0:
                    continue
                t = pd.DataFrame({
                    SYMBOL: np.full(len(ts), sym, dtype=object),
                    "price": price, "volume": vol, "turnover": price * vol,
                    "side": side, "is_cancel": False,
                }, index=ts.floor("min"))
                tm = t.index.time
                t["is_open_auction"] = (
                    (tm >= _OPEN_AUCTION_MIN) & (tm < _OPEN_AUCTION_END))
                frames.append(t)

        if n_ts_total:
            logger.info(
                "逐笔成交: 乱序 %d/%d 行（%.4f%%）自动重排；异常单过滤 %d 行",
                n_ts_reorder, n_ts_total, 100.0 * n_ts_reorder / n_ts_total,
                n_orders_bad)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

    # ------------------------------------------------------------------
    # 逐笔委托
    # ------------------------------------------------------------------

    def load_tick_orders(self, symbols: List[str], start: str,
                         end: str) -> pd.DataFrame:
        """逐笔委托 → 标准委托表（含 09:15 起集合竞价委托）。

        输出列：[symbol, price, volume, side, action]
        - side：1=买委托 / -1=卖委托（委托代码 B/S）
        - action：1=委托 / 0=撤单（委托类型 A/D）
        - 时间戳 floor 到分钟（毫秒精度保留于原始序列排序中）
        """
        frames: List[pd.DataFrame] = []
        n_bad = 0
        for sym in symbols:
            for f in _quote_paths(self.root, sym, "逐笔委托", start, end):
                df = self._read_parquet(f, "逐笔委托")
                if df is None or df.empty:
                    continue
                df = self._clean_raw(df, f.name)
                try:
                    ts = self._file_timestamp(df)
                    price = (pd.to_numeric(df["委托价格"], errors="coerce")
                             .to_numpy() / PRICE_SCALE)
                    vol = pd.to_numeric(df["委托数量"], errors="coerce").to_numpy()
                    side = np.select([df["委托代码"].astype(str) == "B",
                                      df["委托代码"].astype(str) == "S"],
                                     [1, -1], default=0)
                    action = np.select([df["委托类型"].astype(str) == "A",
                                        df["委托类型"].astype(str) == "D"],
                                       [1, 0], default=-1)
                except KeyError as e:
                    logger.warning("委托文件 %s 缺列 %s，跳过", f.name, e)
                    continue
                bad = (price <= 0) | (vol <= 0) | (side == 0) | (action < 0)
                nb = int(bad.sum())
                n_bad += nb
                if nb:
                    price, vol, side, action = (
                        price[~bad], vol[~bad], side[~bad], action[~bad])
                    ts = ts[~bad]
                if len(ts) == 0:
                    continue
                frames.append(pd.DataFrame({
                    SYMBOL: np.full(len(ts), sym, dtype=object),
                    "price": price, "volume": vol,
                    "side": side, "action": action,
                }, index=ts.floor("min")))
        if n_bad:
            logger.warning("逐笔委托异常行（价/量≤0 或标志非法）共 %d 行已过滤", n_bad)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _read_parquet(f: Path, kind: str) -> Optional[pd.DataFrame]:
        """健壮读取：单文件损坏不中断全链路。"""
        try:
            return pd.read_parquet(f)
        except Exception as e:  # noqa: BLE001 - 文件级容错
            logger.warning("%s 文件 %s 读取失败（%s），跳过", kind, f.name, e)
            return None

    @staticmethod
    def _clean_raw(df: pd.DataFrame, fname: str) -> pd.DataFrame:
        """过滤非法自然日/时间行。

        数据商原始文件常在每个文件末尾附加 1 行「自然日/时间=0」的
        收盘汇总脏数据；此类行无时间语义，若进入 _file_timestamp 会
        使 to_datetime 抛错。这里统一在解析前过滤并告警（不中断）。
        """
        if "自然日" not in df.columns:
            return df
        day = pd.to_numeric(df["自然日"], errors="coerce")
        tm = pd.to_numeric(df["时间"], errors="coerce") \
            if "时间" in df.columns else day
        bad = day.isna() | (day <= 0) | tm.isna() | (tm <= 0)
        n = int(bad.sum())
        if n:
            logger.warning("%s 非法自然日/时间 %d 行（置 0/空），已过滤",
                           fname, n)
            return df[~bad]
        return df

    @staticmethod
    def _file_timestamp(df: pd.DataFrame) -> pd.DatetimeIndex:
        """自然日 + 时间 → DatetimeIndex（毫秒精度，强制 ns 单位）。

        pandas 3.0 的 to_datetime 字符串解析返回类型随输入列 dtype 浮动
        （Series 或 DatetimeIndex），这里显式归一为 DatetimeIndex，
        调用方统一使用 Index.floor() 而无需区分 Series.dt 语法。
        """
        ts = (pd.to_datetime(df["自然日"].astype(str), format="%Y%m%d")
              + _hmss_to_timedelta(df["时间"]))
        return pd.DatetimeIndex(ts).astype("datetime64[ns]")
