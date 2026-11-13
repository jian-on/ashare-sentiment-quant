# -*- coding: utf-8 -*-
"""
backtest.py — point-in-time 截面存档 + purged walk-forward 验证骨架

2026-11-13 新增（原 README 标记为「待开发」）。存在的理由：
config.py 里的 COMMISSION / SLIPPAGE / TRAIN_DAYS / VALID_DAYS / OOS_DAYS
长期只是声明，没有任何代码引用；此前项目是「实时横截面观测终端」，
不具备任何样本外证据。没有本模块，任何权重/阈值调整都是不可证伪的玄学。

本模块做三件事：
  1. archive  — 每次运行把当日完整横截面（因子 z、composite、状态、舆情源）落盘为不可变快照
  2. label    — T+5 之后用项目自有腾讯K线补未来 5 日超额收益标签（不依赖付费 API）
  3. report   — 按时序切 TRAIN/VALID/OOS + 5日 embargo，输出 Rank IC / ICIR 与扣费净超额

⚠️ 诚实边界：需要积累 ≥ TRAIN+VALID+OOS+EMBARGO = 245 个交易日截面才有统计意义。
   数据不足时本模块**拒绝给出任何结论**，只报告 INSUFFICIENT_DATA。
   阈值/权重的定参只允许在 TRAIN+VALID 做，OOS 只看一次即冻结。

用法：
  python backtest.py --archive            # 收盘后运行，存档当日横截面
  python backtest.py --label 20261113     # 该日 T+5 过后补标签
  python backtest.py --status             # 查看数据准备度
  python backtest.py --report             # 出样本外报告（数据不足会明确拒绝）
"""
import argparse
import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd

import config as C
import data_loader as dl
import factors as F
import market_sentiment as MS
import recommend as RC
import regime as RG
import sources as SRC

PIT_DIR = os.path.join(C.CACHE_DIR, "pit")
REQUIRED_DAYS = C.TRAIN_DAYS + C.VALID_DAYS + C.OOS_DAYS + C.EMBARGO_DAYS


def _ensure_dir():
    os.makedirs(PIT_DIR, exist_ok=True)


def _path(date_str: str) -> str:
    return os.path.join(PIT_DIR, f"{date_str}.parquet")


# ---------------------------------------------------------------- 1. 存档
def archive_cross_section(date_str: str | None = None, pool_size: int = 300) -> str:
    """
    把当日完整候选横截面落盘。这是 point-in-time 不可变存档：
    一旦写入不再修改因子值，只追加未来收益标签，保证回测不会用到「当时不可得」的信息。
    """
    _ensure_dir()
    date_str = date_str or dt.date.today().strftime("%Y%m%d")
    path = _path(date_str)
    if os.path.exists(path):
        print(f"[存档] {date_str} 已存在，跳过（不可变）：{path}")
        return path

    asof = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    spot = dl.get_spot()
    uni = dl.filter_universe(spot)
    pool = dl.build_stock_pool(uni, max_pool=pool_size)
    thermo = MS.market_thermometer(spot=spot)
    weights, state, _ = RG.resolve_weights(thermo.get("detail", {}))

    pre = F.score_pool(pool, spot, market_score=thermo["market_score"], weights=weights)
    n_cov = min(len(pre), C.SENTIMENT_COVERAGE_N if C.SENTIMENT_COVERAGE_N > 0 else len(pre))
    codes = [str(c) for c in pre.head(n_cov)["code"]]
    sent_map, sent_detail = RC._collect_sentiment(codes)
    cand_pool = {c: pool[c] for c in codes if c in pool}
    scored, diag = F.score_pool(cand_pool, spot, sentiment_map=sent_map or None,
                                market_score=thermo["market_score"], weights=weights,
                                return_diag=True)

    df = scored.copy()
    df["date"] = date_str
    df["asof"] = asof
    df["market_score"] = thermo["market_score"]
    df["state"] = state
    df["weights_effective"] = json.dumps(diag["weights_effective"], ensure_ascii=False)
    df["sentiment_source"] = df["code"].map(
        {c: v.get("source", "UNKNOWN") for c, v in sent_detail.items()})
    df["fwd_ret_5d"] = np.nan        # 标签位，T+5 后由 label 填充
    df["fwd_excess_5d"] = np.nan
    df.to_parquet(path, index=False)
    print(f"[存档] {date_str} 已写入 {len(df)} 只 / asof {asof} / 状态 {state} -> {path}")
    return path


# ---------------------------------------------------------------- 2. 标签
def label_forward_returns(date_str: str, horizon: int = 5) -> int:
    """
    用项目自有腾讯前复权K线补 T+1~T+horizon 收益标签，零新增依赖、零付费 API。
    超额 = 个股收益 - 当日候选池等权平均收益（横截面相对口径，与模型定位一致）。
    T+horizon 尚未到来时明确报 pending，不填假数据。
    """
    path = _path(date_str)
    if not os.path.exists(path):
        raise FileNotFoundError(f"无 {date_str} 存档，请先 --archive")
    df = pd.read_parquet(path)
    if df["fwd_excess_5d"].notna().all():
        print(f"[标签] {date_str} 已完成，跳过")
        return 0

    t0 = pd.to_datetime(date_str, format="%Y%m%d")
    rets = {}
    for code in df["code"].astype(str):
        hist = SRC.fetch_kline_tx(code, days=30)
        if hist is None or hist.empty:
            continue
        h = hist.copy()
        h["date"] = pd.to_datetime(h["date"])
        pos = h.index[h["date"] >= t0]
        if len(pos) == 0:
            continue
        i = pos[0]
        if i + horizon >= len(h):
            continue
        c0 = float(h.loc[i, "close"])
        c1 = float(h.loc[i + horizon, "close"])
        if c0 > 0:
            rets[code] = c1 / c0 - 1

    if len(rets) < max(10, int(len(df) * 0.6)):
        print(f"[标签] {date_str} 仅取到 {len(rets)}/{len(df)} 只，"
              f"T+{horizon} 可能尚未到来，保持 pending")
        return 0
    bench = float(np.mean(list(rets.values())))
    df["fwd_ret_5d"] = df["code"].astype(str).map(rets)
    df["fwd_excess_5d"] = df["fwd_ret_5d"] - bench
    df.to_parquet(path, index=False)
    print(f"[标签] {date_str} 完成 {len(rets)} 只 / 基准收益 {bench:+.4f}")
    return len(rets)


# ---------------------------------------------------------------- 3. 评价
def _rank_ic(x: pd.Series, y: pd.Series) -> float | None:
    a, b = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
    ok = a.notna() & b.notna()
    if int(ok.sum()) < 10:
        return None
    return float(a[ok].rank().corr(b[ok].rank()))


def _ic_stats(ics: list[float]) -> dict:
    a = np.asarray([v for v in ics if v is not None and np.isfinite(v)], dtype=float)
    if len(a) < 3:
        return {"n": len(a), "ic_mean": None, "icir": None, "ic_pos_rate": None,
                "t_stat": None, "p_approx": None}
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    icir = mean / sd if sd > 0 else 0.0
    t = icir * math.sqrt(len(a))
    p = math.erfc(abs(t) / math.sqrt(2))     # 正态近似双尾 p 值
    return {"n": len(a), "ic_mean": round(mean, 4), "icir": round(icir, 4),
            "ic_pos_rate": round(float((a > 0).mean()), 4),
            "t_stat": round(t, 3), "p_approx": round(p, 4)}


def _net_excess(seg: pd.DataFrame, top_n: int) -> dict:
    """Top N 等权的扣费净超额（对齐横截面排序目标，非绝对收益）"""
    vals = []
    for _, g in seg.groupby("date"):
        if g["fwd_excess_5d"].isna().any():
            continue
        top = g.nlargest(min(top_n, len(g)), "composite")
        if len(top) == 0:
            continue
        gross = float(top["fwd_excess_5d"].mean())
        # 买入+卖出各一次：佣金/滑点双边，印花税卖出单边
        cost = 2 * (C.COMMISSION + C.SLIPPAGE) + C.STAMP_TAX
        vals.append(gross - cost)
    if not vals:
        return {"n": 0, "net_excess_mean": None}
    return {"n": len(vals), "net_excess_mean": round(float(np.mean(vals)), 5),
            "net_excess_std": round(float(np.std(vals, ddof=1)), 5) if len(vals) > 1 else None}


def load_panel() -> pd.DataFrame:
    _ensure_dir()
    files = sorted(f for f in os.listdir(PIT_DIR) if f.endswith(".parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(os.path.join(PIT_DIR, f)) for f in files],
                     ignore_index=True).sort_values("date").reset_index(drop=True)


def status() -> dict:
    p = load_panel()
    dates = sorted(p["date"].unique()) if len(p) else []
    labeled = int(p.groupby("date")["fwd_excess_5d"].apply(lambda s: s.notna().all()).sum()) if dates else 0
    return {"存档天数": len(dates), "已补标签天数": labeled,
            "需要天数": REQUIRED_DAYS,
            "最早": dates[0] if dates else None, "最新": dates[-1] if dates else None,
            "ready": labeled >= REQUIRED_DAYS}


def report() -> dict:
    """
    时序 purged walk-forward：TRAIN / VALID / OOS，段间 5 日 embargo 防标签重叠泄漏。
    OOS 只用于最终判定，不参与任何阈值或权重调整。
    """
    st = status()
    if not st["ready"]:
        return {"verdict": "INSUFFICIENT_DATA",
                "reason": f"需 ≥{REQUIRED_DAYS} 个已标注交易日，当前仅 {st['已补标签天数']} 个；"
                          f"本模块拒绝在数据不足时给出任何有效性结论",
                **st}
    p = load_panel()
    p = p[p["fwd_excess_5d"].notna()].copy()
    dates = sorted(p["date"].unique())
    n = len(dates)
    # 三段 + 段间 embargo：留出 2*EMBARGO_DAYS 天不参与定参/评价
    e = C.EMBARGO_DAYS
    tr_end = C.TRAIN_DAYS
    va_end = tr_end + e + C.VALID_DAYS
    oos_start = va_end + e
    segments = {
        "TRAIN": dates[:tr_end],
        "VALID": dates[tr_end + e:va_end],
        "OOS": dates[oos_start:oos_start + C.OOS_DAYS],
    }
    out = {"verdict": None, "segments": {}, "state_layered_ic": {}}
    for name, ds in segments.items():
        seg = p[p["date"].isin(ds)]
        ics = [_rank_ic(g["composite"], g["fwd_excess_5d"]) for _, g in seg.groupby("date")]
        out["segments"][name] = {**_ic_stats(ics), **_net_excess(seg, C.TOP_N)}

    oos = out["segments"]["OOS"]
    # 判定阈值（全部满足才算通过，否则回退基准权重）
    ok = (oos.get("ic_mean") is not None and oos["ic_mean"] > 0
          and oos["icir"] is not None and oos["icir"] >= 0.30
          and oos["ic_pos_rate"] is not None and oos["ic_pos_rate"] >= 0.55
          and oos["p_approx"] is not None and oos["p_approx"] < 0.05
          and oos.get("net_excess_mean") is not None and oos["net_excess_mean"] > 0)
    out["verdict"] = "PASS_OOS" if ok else "FAIL_REVERT_TO_BASELINE"
    out["criteria"] = "OOS: IC>0 且 ICIR>=0.30 且 IC>0占比>=55% 且 p<0.05 且扣费净超额>0"

    # 分市场状态检验（factors.py 顶部风险提示明确要求）
    for state, g in p[p["date"].isin(segments["OOS"])].groupby("state"):
        ics = [_rank_ic(x["composite"], x["fwd_excess_5d"]) for _, x in g.groupby("date")]
        s = _ic_stats(ics)
        out["state_layered_ic"][str(state)] = s
        if s["n"] < C.MIN_STATE_SAMPLES:
            out["state_layered_ic"][str(state)]["note"] = \
                f"样本 <{C.MIN_STATE_SAMPLES}，该状态不得单独定参，回退基准权重"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", action="store_true", help="存档当日横截面")
    ap.add_argument("--label", metavar="YYYYMMDD", help="补未来5日收益标签")
    ap.add_argument("--status", action="store_true", help="查看数据准备度")
    ap.add_argument("--report", action="store_true", help="样本外报告")
    ap.add_argument("--pool", type=int, default=300)
    a = ap.parse_args()
    if a.archive:
        archive_cross_section(pool_size=a.pool)
    if a.label:
        label_forward_returns(a.label)
    if a.status or not any([a.archive, a.label, a.report]):
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    if a.report:
        print(json.dumps(report(), ensure_ascii=False, indent=2))
