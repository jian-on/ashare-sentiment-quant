# -*- coding: utf-8 -*-
"""
backtest.py — point-in-time 截面存档 + purged walk-forward 验证骨架

2026-11-13 新增（原 README 标记为「待开发」）。存在的理由：
config.py 里的 COMMISSION / SLIPPAGE / TRAIN_DAYS / VALID_DAYS / OOS_DAYS
长期只是声明，没有任何代码引用；此前项目是「实时横截面观测终端」，
不具备任何样本外证据。没有本模块，任何权重/阈值调整都是不可证伪的玄学。

本模块做三件事：
  1. archive  — 每次运行把当日完整横截面（因子 z、composite、状态、舆情源）落盘为不可变快照
  2. label    — T+5 之后用项目自有腾讯K线补未来 5 日收益标签（不依赖付费 API）
               买入口径为 T+1 开盘（清单 T 日收盘后才生成），T+1 一字涨停剔除；
               双基准：候选池等权（衡量排序能力）+ 中证全指固定外部基准（衡量真实超额）
  3. report   — 按时序切 TRAIN/VALID/OOS + 5日 embargo，输出 Rank IC / ICIR、
               相对外部基准的扣费净超额，以及「选池贡献 / 打分贡献 / 总超额」三段归因；
               另按 10cm/20cm 板块分层检验 IC 是否同向（只诊断，不自动调权）
  4. quality  — 每日数据质量留痕（asof 一致性、失效因子、舆情覆盖率、板块分布），
               供发现「横截面残缺导致 z 失真」这类静默问题

收盘闸门（2026-11-16）：早于 ARCHIVE_MIN_HHMM 一律拒绝写 PIT 存档，防止盘中
未定型日K被固化进不可变截面。定时任务固定 15:50，正常路径不受影响；补标签不受闸门限制。

⚠️ 诚实边界：需要积累 ≥ TRAIN+VALID+OOS+EMBARGO = 245 个交易日截面才有统计意义。
   数据不足时本模块**拒绝给出任何结论**，只报告 INSUFFICIENT_DATA。
   阈值/权重的定参只允许在 TRAIN+VALID 做，OOS 只看一次即冻结。

用法：
  python backtest.py --daily              # ★盘后一站式：交易日判定→存档→补标签→状态（定时任务用）
  python backtest.py --archive            # 收盘后运行，存档当日横截面
  python backtest.py --label 20261113     # 该日 T+5 过后补标签
  python backtest.py --label-pending      # 扫描并补齐所有可补标签的存档
  python backtest.py --status             # 查看数据准备度
  python backtest.py --quality [N]        # 最近 N 日数据质量摘要（默认 5）
  python backtest.py --report             # 出样本外报告（数据不足会明确拒绝）

日志：data_cache/pit/daily_run.log（追加，UTF-8；目录已被 .gitignore 排除）
"""
import argparse
import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd
import requests

import config as C
import data_loader as dl
import factors as F
import market_sentiment as MS
import recommend as RC
import regime as RG
import sources as SRC

PIT_DIR = os.path.join(C.CACHE_DIR, "pit")
LOG_PATH = os.path.join(PIT_DIR, "daily_run.log")
REQUIRED_DAYS = C.TRAIN_DAYS + C.VALID_DAYS + C.OOS_DAYS + C.EMBARGO_DAYS
# 一字涨停买不到、K线缺失都会让个别标的没有标签。用覆盖率而非「全部非空」
# 判定某日是否已完成标注，否则只要有一只缺失就会永远 pending、每天重复拉取。
LABEL_COVERAGE_MIN = 0.80


def _board_of(code: str) -> str:
    """按代码前缀标记涨跌幅板：20cm（创业板/科创板）与 10cm（主板）。

    ST 股已在 filter_universe 剔除（5cm 不退化为 10cm），北交所也不在候选池内。
    这里只做分层诊断标签，不参与任何打分。
    """
    return "20cm" if str(code).zfill(6).startswith(C.BOARD_20CM_PREFIXES) else "10cm"


def _ensure_dir():
    os.makedirs(PIT_DIR, exist_ok=True)


def _log(msg: str):
    """同时输出到 stdout（供定时任务采集）与盘后日志（供事后排查）"""
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        _ensure_dir()
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _path(date_str: str) -> str:
    return os.path.join(PIT_DIR, f"{date_str}.parquet")


# ---------------------------------------------------------------- 1. 存档
def archive_cross_section(date_str: str | None = None, pool_size: int = 300,
                          force: bool = False) -> str:
    """
    把当日完整候选横截面落盘。这是 point-in-time 不可变存档：
    一旦写入不再修改因子值，只追加未来收益标签，保证回测不会用到「当时不可得」的信息。

    force=True 才允许在收盘闸门之前写入（仅供盘中复盘，风险自负）；
    默认拒绝，因为盘中日K末根未定型，写进去就永久污染这一天的截面。
    """
    _ensure_dir()
    if not force and not is_after_close():
        raise RuntimeError(
            f"当前 {dt.datetime.now():%H:%M} 早于收盘闸门 {C.ARCHIVE_MIN_HHMM}，"
            f"拒绝写 PIT 存档（盘中日K末根未定型，会永久污染当日截面）；"
            f"请在收盘后重跑，或显式加 --force 用于盘中复盘")
    date_str = date_str or dt.date.today().strftime("%Y%m%d")
    path = _path(date_str)
    if os.path.exists(path):
        _log(f"[存档] {date_str} 已存在，跳过（不可变）：{path}")
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
    sent_map, sent_detail = RC._collect_sentiment(codes, archive_date=date_str, asof=asof)
    source_map = {c: str(v.get("score_source", "UNKNOWN")) for c, v in sent_detail.items()}
    cand_pool = {c: pool[c] for c in codes if c in pool}
    scored, diag = F.score_pool(cand_pool, spot, sentiment_map=sent_map or None,
                                sentiment_source_map=source_map or None,
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
    df["sentiment_score_source"] = df["code"].map(source_map)
    df["fwd_ret_5d"] = np.nan        # 标签位，T+5 后由 label 填充
    df["fwd_excess_5d"] = np.nan     # 相对候选池等权（衡量排序能力，IC 用）
    df["fwd_excess_mkt_5d"] = np.nan  # 相对固定外部基准（衡量真实超额，净超额用）
    df["pool_ret_5d"] = np.nan
    df["mkt_ret_5d"] = np.nan
    df["benchmark_symbol"] = C.BENCHMARK_SYMBOL
    # v3 新增：板块分层标签 + 当日数据质量留痕。不改动任何因子值，只增加可观测性。
    df["board"] = df["code"].astype(str).map(_board_of)
    df["quality_json"] = json.dumps(diag, ensure_ascii=False, default=str)
    # 口径版本：因子口径变更时旧存档必须被排除，绝不「旧因子值 + 新收益标签」混算
    df["pit_schema_version"] = C.PIT_SCHEMA_VERSION
    df.to_parquet(path, index=False)
    board_mix = df["board"].value_counts().to_dict()
    _log(f"[存档] {date_str} 已写入 {len(df)} 只 / asof {asof} / 状态 {state} "
         f"/ 板块 {board_mix} -> {path}")
    return path


# ---------------------------------------------------------------- 2. 标签
def fetch_benchmark_return(date_str: str, horizon: int = 5) -> tuple[float | None, dict]:
    """
    固定外部基准（默认中证全指 sh000985）的 T+1 开盘 → T+horizon 收盘收益。

    为什么需要：原 fwd_excess_5d 减的是「当日候选池等权」，而候选池是模型自己选出来的，
    基准随选样漂移，超额收益的含义不稳定，也无法区分「选池赚的」和「排序赚的」。
    指数与个股标签同口径、不受本项目选样影响，是唯一可跨日比较的锚。
    """
    sym = C.BENCHMARK_SYMBOL
    info = {"symbol": sym, "label": C.BENCHMARK_LABEL, "n_bars": 0}
    try:
        url = getattr(SRC, "_working_url", {}).get("url", SRC.TX_KLINE_URLS[0])
        r = requests.get(url, params={"param": f"{sym},day,,,30,qfq"}, timeout=12,
                         headers={**SRC._UA, "Referer": "https://gu.qq.com/"})
        d = r.json()
        node = (d.get("data") or {}).get(sym) or {}
        k = node.get("qfqday") or node.get("day") or []
        if not k:
            info["error"] = "指数K线为空"
            return None, info
        h = pd.DataFrame([{"date": x[0], "open": float(x[1]), "close": float(x[2])}
                          for x in k if len(x) >= 3])
        h["date"] = pd.to_datetime(h["date"])
        t0 = pd.to_datetime(date_str, format="%Y%m%d")
        pos = h.index[h["date"] >= t0]
        info["n_bars"] = len(h)
        if len(pos) == 0 or pos[0] + horizon >= len(h):
            info["error"] = f"T+{horizon} 尚未到来"
            return None, info
        i = pos[0]
        buy, sell = float(h.loc[i + 1, "open"]), float(h.loc[i + horizon, "close"])
        if buy <= 0:
            info["error"] = "指数开盘价异常"
            return None, info
        info.update({"buy_date": str(h.loc[i + 1, "date"].date()),
                     "sell_date": str(h.loc[i + horizon, "date"].date()),
                     "buy": round(buy, 3), "sell": round(sell, 3)})
        return sell / buy - 1, info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info


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
    if "fwd_excess_mkt_5d" in df.columns and len(df) \
            and df["fwd_excess_mkt_5d"].notna().mean() >= LABEL_COVERAGE_MIN:
        _log(f"[标签] {date_str} 已完成，跳过")
        return 0

    t0 = pd.to_datetime(date_str, format="%Y%m%d")
    rets = {}
    n_unbuyable = 0
    for code in df["code"].astype(str):
        hist = SRC.fetch_kline_tx(code, days=30)
        if hist is None or hist.empty:
            continue
        # 必须重置为连续整数索引：下面用 i+horizon 定位 T+5，索引不连续会取错行
        h = hist.copy().reset_index(drop=True)
        h["date"] = pd.to_datetime(h["date"])
        pos = h.index[h["date"] >= t0]
        if len(pos) == 0:
            continue
        i = pos[0]
        if i + horizon >= len(h):
            continue
        # 可执行口径：清单在 T 日收盘后才生成 → 最早 T+1 开盘买入。
        # 用 T 日收盘价当成本等于假设能回到过去成交，会系统性高估收益。
        # 停牌日不在K线里，i+1 自然落到下一个可交易日开盘，符合实际。
        prev_close = float(h.loc[i, "close"])
        buy_row = h.loc[i + 1]
        buy = float(buy_row["open"])
        sell = float(h.loc[i + horizon, "close"])
        if buy <= 0 or prev_close <= 0:
            continue
        # T+1 一字涨停开盘（无振幅）实际买不到，剔除而非假装成交
        limit = 0.198 if str(code).startswith(("300", "301", "688")) else 0.098
        if (float(buy_row["high"]) == float(buy_row["low"])
                and buy / prev_close - 1 >= limit - 0.005):
            n_unbuyable += 1
            continue
        rets[code] = sell / buy - 1

    if len(rets) < max(10, int(len(df) * LABEL_COVERAGE_MIN)):
        _log(f"[标签] {date_str} 仅取到 {len(rets)}/{len(df)} 只，"
             f"T+{horizon} 可能尚未到来，保持 pending")
        return 0
    bench_mkt, bench_info = fetch_benchmark_return(date_str, horizon)
    if bench_mkt is None:
        _log(f"[标签] {date_str} 外部基准不可用（{bench_info.get('error')}），"
             f"保持 pending；绝不退回池内基准，以免新旧样本口径不一致")
        return 0
    bench = float(np.mean(list(rets.values())))
    df["fwd_ret_5d"] = df["code"].astype(str).map(rets)
    df["fwd_excess_5d"] = df["fwd_ret_5d"] - bench            # 相对候选池：衡量排序能力
    df["fwd_excess_mkt_5d"] = df["fwd_ret_5d"] - bench_mkt    # 相对市场：衡量真实超额
    df["pool_ret_5d"] = bench
    df["mkt_ret_5d"] = bench_mkt
    df["benchmark_symbol"] = bench_info["symbol"]
    df.to_parquet(path, index=False)
    _log(f"[标签] {date_str} 完成 {len(rets)} 只 / 一字涨停剔除 {n_unbuyable} 只"
         f" | 池 {bench:+.4f} / {bench_info['label']} {bench_mkt:+.4f}"
         f"（T+1开盘买 → T+{horizon}收盘卖）")
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
    """
    Top N 等权的扣费净超额（相对固定外部基准）+ 三段归因：
      选池贡献 = 候选池等权 − 市场基准        （选了什么池子）
      打分贡献 = TopN − 候选池等权            （池内排序能力，即 fwd_excess_5d）
      总超额   = 选池贡献 + 打分贡献          （= TopN − 市场基准）
    IC/ICIR 仍用池内相对收益衡量排序能力；净超额必须用市场基准衡量真实组合收益。
    """
    cost = 2 * (C.COMMISSION + C.SLIPPAGE) + C.STAMP_TAX   # 佣金/滑点双边 + 印花税卖出单边
    vals, picks, sorts, totals = [], [], [], []
    for _, g in seg.groupby("date"):
        if "fwd_excess_mkt_5d" not in g.columns:
            continue                       # 老 schema 存档不参与，避免口径混算
        top = g.nlargest(min(top_n, len(g)), "composite")
        if len(top) == 0 or top["fwd_excess_mkt_5d"].notna().mean() < LABEL_COVERAGE_MIN:
            continue
        gross = float(top["fwd_excess_mkt_5d"].mean())
        sort_v = float(top["fwd_excess_5d"].mean())
        pick_v = float(top["pool_ret_5d"].iloc[0]) - float(top["mkt_ret_5d"].iloc[0])
        vals.append(gross - cost)
        picks.append(pick_v)
        sorts.append(sort_v)
        totals.append(gross)
    if not vals:
        return {"n": 0, "net_excess_mean": None,
                "attribution": {"note": "无可用样本或全为老 schema"}}
    attr = {"pool_selection": round(float(np.mean(picks)), 5),
            "cross_section_sort": round(float(np.mean(sorts)), 5),
            "gross_total": round(float(np.mean(totals)), 5),
            "cost_per_period": round(cost, 5),
            "identity_check": round(float(np.mean(picks) + np.mean(sorts)
                                          - np.mean(totals)), 6)}
    return {"n": len(vals), "net_excess_mean": round(float(np.mean(vals)), 5),
            "net_excess_std": round(float(np.std(vals, ddof=1)), 5) if len(vals) > 1 else None,
            "attribution": attr}


def _current_schema_files() -> list[str]:
    """
    只返回当前口径版本的存档文件名。旧版本文件保留在磁盘供审计，但绝不进入
    回测面板、也不再补标签 —— 否则会形成「旧因子值 + 新收益标签」的混杂面板。
    """
    _ensure_dir()
    out = []
    for fn in sorted(f for f in os.listdir(PIT_DIR) if f.endswith(".parquet")):
        try:
            v = pd.read_parquet(os.path.join(PIT_DIR, fn),
                                columns=["pit_schema_version"])["pit_schema_version"].iloc[0]
        except (KeyError, ValueError, IndexError, OSError):
            continue                    # 无版本列 / 空文件 = v1 老存档，排除
        if int(v) == C.PIT_SCHEMA_VERSION:
            out.append(fn)
    return out


def load_panel() -> pd.DataFrame:
    files = _current_schema_files()
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(os.path.join(PIT_DIR, f)) for f in files],
                     ignore_index=True).sort_values("date").reset_index(drop=True)


# ------------------------------------------------- 4. 盘后无人值守流水线
def is_after_close(now: dt.datetime | None = None) -> bool:
    """
    收盘闸门：本机时钟为北京时间，早于 ARCHIVE_MIN_HHMM 一律不得写 PIT 存档。

    为什么不用快照 asof 判断：停牌/僵死股的最后成交时间会停留在早盘，若按 asof
    阻断会重新引入「收盘后误判为盘中」的老 bug（factors.py 已为此改过一次）。
    因此只用本机时钟做硬闸门，asof 仅用于截面一致性诊断。
    """
    now = now or dt.datetime.now()
    hh, mm = (int(x) for x in C.ARCHIVE_MIN_HHMM.split(":"))
    return (now.hour, now.minute) >= (hh, mm)


def spot_trading_day(spot: pd.DataFrame) -> tuple[bool, str | None]:
    """
    用腾讯快照字段 30 的行情日期众数判定「今天是不是交易日」。

    为什么必须判：周末/节假日腾讯仍返回上一交易日快照，若直接用
    date.today() 作存档日期，会把同一份截面存成多个日期，制造重复样本、
    虚增面板天数，最终让 walk-forward 的统计显著性完全失真。
    解析不出来一律返回 False（宁可漏存一天，绝不存错截面）。
    """
    if spot is None or len(spot) == 0 or "asof" not in spot.columns:
        return False, None
    days = spot["asof"].astype(str).str[:8]
    days = days[days.str.fullmatch(r"\d{8}", na=False)]
    if days.empty:
        return False, None
    mode_day = str(days.mode().iloc[0])
    return mode_day == dt.date.today().strftime("%Y%m%d"), mode_day


def label_pending(min_age_days: int = 5) -> int:
    """
    扫描 PIT 存档，把所有「T+5 已过但尚未补标签」的日期补齐，幂等。
    交给脚本自己判断 T+5 是否真的到来（数据不够会保持 pending，次日再试），
    这样定时任务不必推算交易日历，法定长假也不会漏补。
    """
    _ensure_dir()
    today = dt.date.today()
    done = 0
    for fn in _current_schema_files():      # 老口径存档直接跳过，不重复拉 K 线
        date_str = fn[:-len(".parquet")]
        try:
            d = dt.datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            continue
        if (today - d).days < min_age_days:
            continue
        path = os.path.join(PIT_DIR, fn)
        try:
            head = pd.read_parquet(path)
            col = head["fwd_excess_mkt_5d"] if "fwd_excess_mkt_5d" in head.columns else None
        except Exception as e:
            _log(f"[标签] {date_str} 读取失败：{type(e).__name__}: {e}")
            continue
        if col is not None and len(col) and col.notna().mean() >= LABEL_COVERAGE_MIN:
            continue                       # 已完成，幂等跳过
        # 缺 fwd_excess_mkt_5d 的老存档会被重新标注，自动补齐新 schema
        try:
            n = label_forward_returns(date_str)
            if n:
                done += 1
                _log(f"[标签] {date_str} 补齐 {n} 只")
        except Exception as e:
            _log(f"[标签] {date_str} 失败：{type(e).__name__}: {e}")
    return done


def daily_run(pool_size: int = 300) -> int:
    """定时任务唯一入口：交易日判定 → 存档 → 补标签 → 状态。返回进程退出码。"""
    today = dt.date.today().strftime("%Y%m%d")
    _log(f"=== daily_run 开始 {today} ===")
    failed = False
    try:
        spot = dl.get_spot()
    except Exception as e:
        _log(f"[存档] 行情快照获取失败，跳过当日存档：{type(e).__name__}: {e}")
        spot = None
        failed = True

    archive_skipped = None
    if spot is not None:
        is_td, mode_day = spot_trading_day(spot)
        if not is_td:
            _log(f"[存档] 今日非交易日（快照行情日 {mode_day or '无法解析'} ≠ {today}），"
                 f"跳过存档，避免重复截面")
            archive_skipped = "非交易日"
        elif not is_after_close():
            _log(f"[存档] {today} 当前 {dt.datetime.now():%H:%M} 早于收盘闸门 "
                 f"{C.ARCHIVE_MIN_HHMM}，跳过存档（日K末根未定型，写进去会永久污染截面）；"
                 f"请在收盘后重跑 --daily；补标签照常执行")
            archive_skipped = "盘中提前执行"
        else:
            try:
                archive_cross_section(today, pool_size=pool_size)
            except Exception as e:
                _log(f"[存档] {today} 失败：{type(e).__name__}: {e}")
                failed = True
    else:
        archive_skipped = "快照获取失败"

    n = label_pending()                     # 休市日、盘中提前执行时同样可补标签
    st = status()
    if archive_skipped:
        st["本次存档"] = f"跳过（{archive_skipped}）"
    _log("[状态] " + json.dumps(st, ensure_ascii=False))
    try:
        q = quality_summary(days=1)
        if q["alerts"]:
            _log("[质量] " + "；".join(q["alerts"][:3]))
    except Exception as e:
        _log(f"[质量] 摘要生成失败（不影响存档与补标签）：{type(e).__name__}: {e}")
    if not st["ready"]:
        _log(f"[结论] 数据不足（{st['已补标签天数']}/{st['需要天数']} 个已标注交易日），"
             f"拒绝给出任何有效性结论；本次仅完成存档与标签补齐")
    _log(f"=== daily_run 结束 | 标签补齐 {n} 天 | 失败={'是' if failed else '否'} ===")
    return 1 if failed else 0


def status() -> dict:
    p = load_panel()
    dates = sorted(p["date"].unique()) if len(p) else []
    # 只有带「市场超额」列的存档才算已标注；老 schema 必须重新标注，避免口径混算
    labeled = int(p.groupby("date")["fwd_excess_mkt_5d"]
                  .apply(lambda s: s.notna().mean() >= LABEL_COVERAGE_MIN).sum()) \
        if dates and "fwd_excess_mkt_5d" in p.columns else 0
    return {"存档天数": len(dates), "已补标签天数": labeled,
            "需要天数": REQUIRED_DAYS,
            "最早": dates[0] if dates else None, "最新": dates[-1] if dates else None,
            "ready": labeled >= REQUIRED_DAYS}


def quality_summary(days: int = 5) -> dict:
    """最近 N 日数据质量摘要：asof 一致性、有效权重、失效因子、舆情覆盖、板块分布。

    这些字段在存档时由 score_pool 的 diag 原样写入 quality_json，不重算、不回溯修改，
    因此可以用来发现「K线残缺 → 横截面 z 失真」这类不会报错但会毁掉证据的静默问题。
    """
    files = _current_schema_files()
    if not files:
        return {"n_days": 0, "note": "无当前版本存档（v1 旧存档不参与质量检查）"}
    out, alerts = [], []
    for fn in files[-max(1, days):]:
        try:
            df = pd.read_parquet(os.path.join(PIT_DIR, fn))
            q = json.loads(df["quality_json"].iloc[0]) if "quality_json" in df else {}
        except (KeyError, ValueError, TypeError, IndexError, OSError) as e:
            alerts.append(f"{fn} 质量记录读取失败：{type(e).__name__}")
            continue
        rec = {"date": fn[:-len(".parquet")], "n_stocks": len(df),
               "board": df["board"].value_counts().to_dict() if "board" in df else {},
               "asof": q.get("asof"), "asof_ok": q.get("asof_ok"),
               "asof_span_sec": q.get("asof_span_sec"),
               "effective_weight_total": q.get("effective_weight_total"),
               "dropped_factors": q.get("dropped_factors") or [],
               "sentiment_coverage": q.get("sentiment_coverage"),
               "sentiment_sources": q.get("sentiment_sources") or {}}
        out.append(rec)
        if rec["asof_ok"] is False:
            alerts.append(f"{rec['date']} 截面时点不一致（跨度 {rec['asof_span_sec']}s）")
        if rec["effective_weight_total"] is not None and rec["effective_weight_total"] < 0.999:
            alerts.append(f"{rec['date']} 有效权重仅 {rec['effective_weight_total']}")
        if rec["dropped_factors"]:
            alerts.append(f"{rec['date']} 失效因子 {rec['dropped_factors']}")
        if rec["sentiment_coverage"] is not None and rec["sentiment_coverage"] < 0.5:
            alerts.append(f"{rec['date']} 舆情覆盖率仅 {rec['sentiment_coverage']}")
    return {"n_days": len(out), "recent": out, "alerts": alerts[:20]}


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
    need = ["fwd_excess_5d", "fwd_excess_mkt_5d", "pool_ret_5d", "mkt_ret_5d"]
    missing = [c for c in need if c not in p.columns]
    if missing:
        return {"verdict": "SCHEMA_MISMATCH",
                "reason": f"面板缺少 {missing}；请用当前代码重新标注后再出报告", **st}
    p = p[p["fwd_excess_5d"].notna() & p["fwd_excess_mkt_5d"].notna()].copy()
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
    out = {"verdict": None, "segments": {}, "state_layered_ic": {},
           "board_layered_ic": {}, "board_layered_excess": {},
           "benchmark": {"symbol": C.BENCHMARK_SYMBOL, "label": C.BENCHMARK_LABEL}}
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
    out["criteria"] = ("OOS: IC>0 且 ICIR>=0.30 且 IC>0占比>=55% 且 p<0.05 "
                       "且相对固定外部基准的扣费净超额>0；IC 用池内相对收益衡量排序，"
                       "净超额用市场基准衡量真实收益；attribution 拆选池/打分贡献")

    # 分市场状态检验（factors.py 顶部风险提示明确要求）
    for state, g in p[p["date"].isin(segments["OOS"])].groupby("state"):
        ics = [_rank_ic(x["composite"], x["fwd_excess_5d"]) for _, x in g.groupby("date")]
        s = _ic_stats(ics)
        out["state_layered_ic"][str(state)] = s
        if s["n"] < C.MIN_STATE_SAMPLES:
            out["state_layered_ic"][str(state)]["note"] = \
                f"样本 <{C.MIN_STATE_SAMPLES}，该状态不得单独定参，回退基准权重"

    # 分板块检验（2026-11-16）：10cm/20cm 涨跌幅不同，low_vol 与量能因子分布会混合。
    # 只诊断 IC 是否同向，不据此自动调权 —— 自由度增加必须有样本外证据才允许。
    if "board" in p.columns:
        for board, g in p[p["date"].isin(segments["OOS"])].groupby("board"):
            ics = [_rank_ic(x["composite"], x["fwd_excess_5d"]) for _, x in g.groupby("date")]
            s = _ic_stats(ics)
            out["board_layered_ic"][str(board)] = s
            out["board_layered_excess"][str(board)] = _net_excess(g, C.TOP_N)
            if s["n"] < C.MIN_BOARD_SAMPLES:
                out["board_layered_ic"][str(board)]["note"] = \
                    f"样本 <{C.MIN_BOARD_SAMPLES}，该板块不单独定参；"
                out["board_layered_ic"][str(board)]["note"] += \
                    "两板块 IC 反向时须回退基准权重，不得按板块调权"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", action="store_true", help="盘后一站式（定时任务用）")
    ap.add_argument("--archive", action="store_true", help="存档当日横截面")
    ap.add_argument("--label", metavar="YYYYMMDD", help="补未来5日收益标签")
    ap.add_argument("--label-pending", action="store_true", help="补齐所有可补标签")
    ap.add_argument("--status", action="store_true", help="查看数据准备度")
    ap.add_argument("--quality", type=int, nargs="?", const=5, metavar="N",
                    help="最近 N 日数据质量摘要（默认 5）")
    ap.add_argument("--report", action="store_true", help="样本外报告")
    ap.add_argument("--force", action="store_true",
                    help="绕过收盘闸门，允许盘中写存档（仅供复盘，会污染 PIT 证据链）")
    ap.add_argument("--pool", type=int, default=300)
    a = ap.parse_args()
    if a.daily:
        raise SystemExit(daily_run(pool_size=a.pool))
    if a.archive:
        archive_cross_section(pool_size=a.pool, force=a.force)
    if a.label:
        label_forward_returns(a.label)
    if a.label_pending:
        label_pending()
    if a.status or not any([a.archive, a.label, a.label_pending, a.quality is not None,
                            a.report, a.force]):
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    if a.quality is not None:
        print(json.dumps(quality_summary(a.quality), ensure_ascii=False, indent=2))
    if a.report:
        print(json.dumps(report(), ensure_ascii=False, indent=2))
