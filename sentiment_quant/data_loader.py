# -*- coding: utf-8 -*-
"""
data_loader.py — 模块一：标的过滤 + 前复权行情（项目工作顺序第1步，地基）

改动点：
- 全部行情统一前复权（规则1），避免分红除权导致因子跳变
- 四重过滤：ST/退市、停牌、上市<60日、日均成交额过低
- 带本地缓存（规则7）

【数据源架构（2026-11-05 重构，根治东财限流）】
主源 = sources.py 腾讯行情（批量快照 + 并发前复权K线，实测全A 0.7秒 / K线5ms一只）
备源 = 东财 akshare（仅在腾讯失败时兜底，不再是主链路）
原因：东财是逐只拉取，150 只池 = 200+ 请求必然触发 IP 封禁，调参无法根治；
      腾讯支持批量查询，请求数从 200+ 降到 15 次，且无限流。

风险点 / 调试提示：
- 腾讯接口字段顺序若变更 → 运行 `python sources.py --debug` 打印原始结构
- 东财兜底时字段名随 akshare 版本变化 → 运行 `python data_loader.py --debug`
"""
import os
import time
import hashlib
import numpy as np
import pandas as pd

import config as C
import sources as SRC

import akshare as ak

pd.set_option("display.width", 200)


class RateLimited(Exception):
    """东财push2行情接口限流。触发后应立即停止请求等待恢复，切勿重试（会延长限流）。"""


# ---- 全局限流熔断：所有东财请求统一走此节流器 ----
_throttle_state = {"t": 0.0}
_MIN_INTERVAL = C.MIN_REQUEST_INTERVAL  # 秒：两次东财请求最小间隔


def _throttle():
    import time as _t
    wait = _MIN_INTERVAL - (_t.time() - _throttle_state["t"])
    if wait > 0:
        _t.sleep(wait)
    _throttle_state["t"] = _t.time()


def _is_rate_limit(e: Exception) -> bool:
    return "RateLimit" in type(e).__name__ or "限流" in str(e)


# 降级标记：True表示本次返回的是过期缓存（限流降级），供UI层警示
LAST_SPOT_STALE = False


def _cache_path(name: str) -> str:
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    safe = hashlib.md5(name.encode()).hexdigest()[:16]
    return os.path.join(C.CACHE_DIR, f"{name[:40]}_{safe}.parquet")


def _read_cache(name: str, ttl_sec: int | None):
    p = _cache_path(name)
    if os.path.exists(p) and (ttl_sec is None or time.time() - os.path.getmtime(p) < ttl_sec):
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


def _write_cache(name: str, df: pd.DataFrame):
    try:
        df.to_parquet(_cache_path(name))
    except Exception as e:
        print(f"[cache写入失败，不影响主流程] {e}")


def _read_any_cache(prefix: str) -> pd.DataFrame | None:
    """
    按文件名前缀找最新缓存（兼容历史命名规则，避免改key后被迫全量重拉而撞限流）。
    """
    try:
        if not os.path.isdir(C.CACHE_DIR):
            return None
        cands = [f for f in os.listdir(C.CACHE_DIR)
                 if f.startswith(prefix[:40]) and f.endswith(".parquet")]
        if not cands:
            return None
        newest = max(cands, key=lambda f: os.path.getmtime(os.path.join(C.CACHE_DIR, f)))
        return pd.read_parquet(os.path.join(C.CACHE_DIR, newest))
    except Exception:
        return None


def get_spot(ttl_sec: int = None, allow_stale: bool = None) -> pd.DataFrame:
    """
    全A实时快照。核心列：代码/名称/最新价/涨跌幅/成交额(元)/换手率/量比/总市值
    数据源：腾讯批量（主）→ 东财（备）
    口径：严格实时（config.ALLOW_STALE_FALLBACK=False 时不返回任何过期缓存）
    """
    global LAST_SPOT_STALE
    LAST_SPOT_STALE = False
    ttl_sec = C.SPOT_TTL_SEC if ttl_sec is None else ttl_sec
    allow_stale = C.ALLOW_STALE_FALLBACK if allow_stale is None else allow_stale

    df = _read_cache("spot", ttl_sec)
    if df is not None:
        return df

    # ---- 主源：腾讯批量（12-26 次请求，约1秒，无限流风险）----
    try:
        df = SRC.fetch_spot_tx()
        if df is not None and not df.empty:
            _write_cache("spot", df)
            return df
    except Exception as e:
        print(f"[get_spot] 腾讯主源失败 {type(e).__name__}: {e} → 切换东财备源")

    # ---- 备源：东财 ----
    em_rem = SRC.blocked_remain("em")
    if em_rem > 0:
        raise RateLimited(f"主源可用性异常，且东财备源仍在冷却（还需 {em_rem:.0f} 秒）。"
                          f"冷却期内不发请求，以免延长封禁")
    _throttle()
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        if _is_rate_limit(e):
            SRC.mark_blocked("em", 180)
            stale = _read_cache("spot", ttl_sec=None)
            if allow_stale and stale is not None:
                LAST_SPOT_STALE = True
                print("[get_spot] 两个数据源均不可用，降级返回过期快照（数据滞后）")
                return stale
            raise RateLimited("全A实时快照两源均失败（腾讯异常 + 东财限流）。"
                              "严格实时模式下不使用过期缓存，请稍后重试") from e
        raise
    # 防御性类型转换（东财接口偶发把数值列给成字符串）
    for col in ["最新价", "成交额", "换手率", "量比", "60日涨跌幅", "总市值"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _write_cache("spot", df)
    return df


def filter_universe(spot: pd.DataFrame) -> pd.DataFrame:
    """
    标的四重过滤（规则1）。返回通过筛选的候选池快照。
    注意：上市<60日在这一步先用"60日涨跌幅是否为NaN"做粗筛，
    精筛在 fetch_hist 中用真实K线根数兜底（次新股拉不出60根）。
    """
    df = spot.copy()
    n0 = len(df)

    # 1) 代码前缀过滤（4、8、9开头=北交所，不在白名单内即剔除）
    #    2026-11-13 修复：原实现只认 ("60","00") + 可选 688，
    #    创业板 300/301 被静默全部剔除，实盘板块分布里创业板数量为 0。
    code = df["代码"].astype(str)
    prefixes = ["60", "00"]
    if C.INCLUDE_CHINEXT:
        prefixes.append("30")       # 创业板 300/301（20cm，混池风险已在 README 标注）
    if not C.EXCLUDE_688:
        prefixes.append("688")      # 科创板（20cm）
    df = df[code.str.startswith(tuple(prefixes))]
    n1 = len(df)

    # 2) ST/退市
    if C.EXCLUDE_ST:
        name = df["名称"].astype(str)
        df = df[~name.str.contains("ST|退", na=False)]
    n2 = len(df)

    # 3) 停牌：最新价或成交额为空/为0
    df = df[df["最新价"].notna() & (df["最新价"] > 0) & df["成交额"].notna() & (df["成交额"] > 0)]
    n3 = len(df)

    # 4) 低流动性：当日成交额低于阈值（更严谨可用20日均值，此处快照先用当日）
    df = df[df["成交额"] >= C.MIN_AMOUNT]
    n4 = len(df)

    # 5) 次新粗筛：腾讯批量快照不返回「60日涨跌幅」，该列恒缺，此步在腾讯主源下不生效。
    #    真正的次新剔除在 build_stock_pool 按实际K线根数完成（MIN_LISTED_DAYS+5）。
    #    保留列检查仅为兼容未来提供该字段的数据源，不再对外宣称已完成过滤。
    if "60日涨跌幅" in df.columns:
        df = df[df["60日涨跌幅"].notna()]
    n5 = len(df)

    print(f"[标的过滤] 全A {n0} -> 板块 {n1} -> 剔ST {n2} -> 剔停牌 {n3} "
          f"-> 剔低流动性 {n4} -> 可选次新粗筛 {n5}（次新精筛在K线层）")
    return df.reset_index(drop=True)


def _normalize_hist(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount",
        "换手率": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "close", "high", "low", "volume", "amount"]
              + (["turnover"] if "turnover" in df.columns else [])]


def fetch_hist(symbol: str, days: int = None) -> pd.DataFrame | None:
    """
    单只股票前复权日K。
    数据源：腾讯（主，约5ms/只）→ 东财（备）
    - 缓存 TTL = HIST_TTL_SEC（盘中5分钟），过期则重拉
    返回列：date/open/close/high/low/volume[/turnover]
    注意：腾讯源不提供历史换手率，若缺失率过高会删除该列，
          让因子层显式降级为中性值（宁可中性，也不用错位数据）。
    """
    days = days or C.HIST_DAYS
    key = f"hist_{symbol}_qfq"
    cached = _read_cache(key, ttl_sec=C.HIST_TTL_SEC)
    if cached is not None:
        return cached

    old = _read_cache(key, ttl_sec=None)
    if old is None:
        old = _read_any_cache(f"hist_{symbol}_")

    # ---- 主源：腾讯前复权K线 ----
    df = SRC.fetch_kline_tx(symbol, days=days)
    if df is not None and not df.empty:
        if old is not None and not old.empty:
            df = (pd.concat([old, df], ignore_index=True)
                  .drop_duplicates(subset="date", keep="last")
                  .sort_values("date").reset_index(drop=True))
        # 换手率缺失率过高 → 删除该列，让因子层中性降级（不造数据）
        if "turnover" in df.columns and df["turnover"].isna().mean() > 0.5:
            df = df.drop(columns=["turnover"])
        if len(df) >= 25:
            _write_cache(key, df)
            return df

    # ---- 备源：东财 ----
    if SRC.blocked_remain("em") > 0:
        raise RateLimited(f"K线主源异常，东财备源冷却中（{symbol}）。请待冷却结束再试")
    end = pd.Timestamp.now().strftime("%Y%m%d")
    if old is not None and not old.empty:
        start = (old["date"].max() - pd.Timedelta(days=3)).strftime("%Y%m%d")
    else:
        start = (pd.Timestamp.now() - pd.Timedelta(days=int(days * 1.7))).strftime("%Y%m%d")

    _throttle()
    try:
        raw = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                 start_date=start, end_date=end, adjust="qfq")
    except Exception as e:
        if _is_rate_limit(e):
            SRC.mark_blocked("em", 180)
            if C.ALLOW_STALE_FALLBACK and old is not None:
                print(f"[fetch_hist {symbol}] 两源均失败，降级返回过期K线")
                return old
            raise RateLimited(f"K线两源均失败（腾讯 + 东财限流，于{symbol}），已进入冷却") from e
        # 2026-11-13 修复：原实现在此无条件 return old（可能是 11 天前的缓存），
        # 绕过 ALLOW_STALE_FALLBACK 且无任何标记，上层会把过期K线当实时数据打分。
        print(f"[行情获取失败 {symbol}] {type(e).__name__}: {e}")
        if C.ALLOW_STALE_FALLBACK and old is not None:
            print(f"[fetch_hist {symbol}] 严格实时已关闭，显式降级返回过期K线")
            return old
        return None
    if raw is None or raw.empty:
        if C.ALLOW_STALE_FALLBACK and old is not None:
            print(f"[fetch_hist {symbol}] 严格实时已关闭，显式降级返回过期K线")
            return old
        return None

    new = _normalize_hist(raw)
    if old is not None and not old.empty:
        df = (pd.concat([old, new], ignore_index=True)
              .drop_duplicates(subset="date", keep="last")
              .sort_values("date").reset_index(drop=True))
    else:
        df = new
    _write_cache(key, df)
    return df


def build_stock_pool(spot: pd.DataFrame, max_pool: int = 300) -> dict[str, pd.DataFrame]:
    """
    候选池选样（2026-11-13 修订）+ 并发拉前复权K线。
    - 默认 "stratified"：按当日成交额分位分 5 层等额抽样，固定随机种子可复现。
      原「成交额降序 Top N」会把中小盘题材股系统性挤出，实盘基线 Top10 全是
      大盘蓝筹，与题材短线主线冲突；分层抽样让各市值层都有代表。
    - "top_amount"：保留旧行为，但存在「用当日结果选样 + 用当日结果打分」的自我指涉偏差。
    性能：4 线程并发，120 只约 4 秒。
    精筛：K线不足 MIN_LISTED_DAYS+5 根的视为次新/数据异常，剔除。
    返回 {code: qfq_df}
    """
    if C.POOL_SELECT_MODE == "stratified" and len(spot) > max_pool:
        amt = pd.to_numeric(spot["成交额"], errors="coerce").reset_index(drop=True)
        strata = pd.qcut(amt.rank(method="first"), C.POOL_STRATA, labels=False).to_numpy()
        rng = np.random.default_rng(C.POOL_RANDOM_SEED)
        per = max_pool // C.POOL_STRATA
        picked: list[int] = []
        for g in range(C.POOL_STRATA):
            members = np.flatnonzero(strata == g)
            if len(members):
                chosen = rng.choice(members, size=min(per, len(members)), replace=False)
                picked.extend(int(x) for x in chosen)
        taken = set(picked)
        remain = [i for i in range(len(spot)) if i not in taken]
        if len(picked) < max_pool and remain:
            extra = rng.choice(remain, size=min(max_pool - len(picked), len(remain)), replace=False)
            picked.extend(int(x) for x in extra)
        pool = spot.iloc[picked[:max_pool]]
        print(f"[选池] 成交额分 {C.POOL_STRATA} 层等额抽样 "
              f"(seed={C.POOL_RANDOM_SEED}) -> {len(pool)} 只")
    else:
        pool = spot.sort_values("成交额", ascending=False).head(max_pool)
        print(f"[选池] 成交额降序 Top{max_pool}（旧口径，存在当日结果选样偏差）")
    codes = [str(c) for c in pool["代码"].tolist()]

    result: dict[str, pd.DataFrame] = {}
    need: list[str] = []
    for code in codes:
        cached = _read_cache(f"hist_{code}_qfq", ttl_sec=C.HIST_TTL_SEC)
        if cached is not None and len(cached) >= C.MIN_LISTED_DAYS + 5:
            result[code] = cached
        else:
            need.append(code)

    if need:
        try:
            fetched = SRC.fetch_kline_batch(need, days=C.HIST_DAYS)
        except SRC.SourceBlocked as e:
            raise RateLimited(f"腾讯K线源被临时拦截（{e}）。通常1-3分钟自动恢复，"
                              f"期间请勿重复点击") from e
        if not fetched and need:
            raise RuntimeError(f"腾讯K线全部失败（{len(need)}只），请运行 python sources.py --debug 排查")
        if len(fetched) < len(need) * 0.7:
            # 部分失败会静默缩小横截面，winsorize_z 在残缺池上重算 → z 失真
            print(f"[警告] K线覆盖率仅 {len(fetched)}/{len(need)} "
                  f"({len(fetched)/len(need):.0%})，本次横截面 z 分数可能失真，排序结果不宜采信")
        dropped = []
        for code, df in fetched.items():
            if len(df) < C.MIN_LISTED_DAYS + 5:
                dropped.append(code)          # 次新/数据异常
                continue
            _write_cache(f"hist_{code}_qfq", df)
            result[code] = df
        missing = [c for c in need if c not in fetched]
        print(f"[股票池] 缓存命中 {len(codes) - len(need)} | 新拉 {len(fetched)} "
              f"| 剔除(次新/无数据) {len(dropped) + len(missing)}")
    else:
        print(f"[股票池] 全部命中缓存 {len(result)} 只")

    print(f"[股票池] 最终可用 {len(result)} 只")
    return result


if __name__ == "__main__":
    # 调试入口：python data_loader.py
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="打印spot原始列名，用于排查接口字段变化")
    args = ap.parse_args()
    sp = get_spot()
    if args.debug:
        print("spot列名：", list(sp.columns))
        print(sp.head(3))
    uni = filter_universe(sp)
    pool = build_stock_pool(uni, max_pool=20)
    if pool:
        sample_code = next(iter(pool))
        print("\n样例前复权K线（%s）：" % sample_code)
        print(pool[sample_code].tail(5))
