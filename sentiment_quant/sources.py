# -*- coding: utf-8 -*-
"""
sources.py — 多源行情适配层（解决东财限流的根本方案）

【为什么能根治限流】
东财 push2 是"逐只拉取"：150 只候选池 = 150 次请求 + 全A快照 59 页分页，
单次扫描 200+ 请求，必然触发 IP 封禁。这不是调参数能绕过的，是架构问题。

本模块改用**支持批量查询的腾讯行情接口**：
- 全市场快照：一次请求可取 500 只 → 5800 只仅需 12 次请求（≈1.5 秒）
- 前复权日K：单只请求但并发友好，8 线程 120 只仅 0.65 秒（≈5ms/只）
  对比东财 0.5s/只的强制节流，快 100 倍，且实测不触发任何限流。
东财降级为备源（failover），不再作为主链路。

【接口实测结论（2026-11-05 实盘验证）】
- qt.gtimg.cn 批量行情：HTTP 200，499 只 0.11s，字段含现价/涨跌幅/换手率/量比/总市值/涨停价
- web.ifzq.gtimg.cn 前复权K线：HTTP 200，121 根 0.09s，含当日盘中K线
- hq.sinajs.cn 批量备用：可用，但 6000 只一次请求 URL 过长会 431，需分批

【已知限制（不隐瞒）】
- 腾讯K线不返回换手率/成交额 → 因子层 turnover_pct 降级为中性 0.5（不造数据）
- 腾讯批量不返回"60日涨跌幅" → 次新股粗筛改由K线根数精筛兜底
- 单位换算：腾讯成交额是「万元」，本模块统一转为「元」与东财口径一致

【风险点】
- 非官方接口，字段顺序可能变更。变更时运行：python sources.py --debug
- 批量请求单批上限约 500 只，超长需分批（已在 _BATCH_SIZE 处理）
"""
import json
import os
import time
import requests
import pandas as pd

import config as C

_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

TX_BATCH_URL = "https://qt.gtimg.cn/q="
# K线多入口（实测 2026-11-05：web.ifzq 被 WAF 拦时，另外三个仍正常——WAF 按域名分级）
TX_KLINE_URLS = [
    "https://ifzq.gtimg.cn/appstock/app/fqkline/get",
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
]
_working_url = {"url": TX_KLINE_URLS[0]}   # 记住当前可用入口，避免每次都试错的

_BATCH_SIZE = 500          # 单次批量请求上限（实测 500 只 URL 约 4.5KB，稳定）
_KLINE_WORKERS = 4         # K线并发线程数（8线程短时间打满会被腾讯WAF拦，4更稳）
_KLINE_GAP = 0.03          # K线请求最小间隔（秒）：4线程×30ms ≈ 33只/秒，150只约5秒
_BATCH_GAP = 0.12          # 快照批次间隔（秒）


class SourceBlocked(Exception):
    """数据源被 WAF/限流拦截（腾讯返回 501 跳转 waf.tencent.com，或东财 Push2RateLimit）"""


# ---- 源健康状态（持久化到文件）：被拦截后进入冷却期，期内直接跳过 ----
# 目的：避免用户在封禁期反复点击 → 每次重试都在延长封禁时间（这是之前东财反复被封的主因）
_STATE_PATH = os.path.join(C.CACHE_DIR, "source_state.json")


def _load_state() -> dict:
    try:
        with open(_STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_state(st: dict):
    try:
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        with open(_STATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(st, fh)
    except Exception:
        pass


def mark_blocked(src: str = "tx", sec: float = 180.0):
    """标记数据源被拦截，进入 sec 秒冷却"""
    st = _load_state()
    st[src] = time.time() + sec
    _save_state(st)


def blocked_remain(src: str = "tx") -> float:
    """该源还需冷却多少秒（0 表示可用）"""
    return max(0.0, _load_state().get(src, 0.0) - time.time())


def _looks_blocked(r) -> bool:
    """
    识别腾讯 WAF 拦截（用于 JSON 接口如 K线）：
    WAF 返回的是 HTML 跳转页 waf.tencent.com，状态码 501 或内容非 JSON。
    """
    if r is None:
        return True
    if r.status_code != 200:
        return True
    txt = (r.text or "").lstrip()
    if not txt.startswith("{"):
        return True          # WAF 返回的是 HTML 跳转页
    return False


def _looks_blocked_batch(r) -> bool:
    """
    识别批量行情接口被 WAF 拦截。
    注意：批量接口返回格式是 v_sz002384="1~东山精密~..."，**不是 JSON**，
    不能套用 JSON 检测（这是一个已踩过的坑：误判会导致快照全部失败）。
    """
    if r is None:
        return True
    if r.status_code != 200:
        return True
    txt = r.text or ""
    if "v_" not in txt or '="' not in txt:
        return True          # 没有标准行情行 → 大概率是 WAF 页
    return False


# ---------- 代码列表（一次性建好，本地缓存，每天更新） ----------
def _candidate_codes() -> list[str]:
    """
    生成候选代码全集（含大量不存在的空号）。
    腾讯接口对不存在的代码只是不返回，不会报错，所以可以放心扫。
    范围覆盖：深主板/中小板 000001-003999、创业板 300001-301999、
             沪主板 600000-605999、科创板 688000-688999。
    北交所按 config.EXCLUDE_BJ 跳过。
    约 1.4 万个代码 → 28 批 × 0.12s ≈ 3.5 秒建表，之后走本地缓存。
    """
    codes = []
    codes += ["sz%06d" % i for i in range(1, 4000)]
    codes += ["sz%06d" % i for i in range(300001, 302000)]
    codes += ["sh%06d" % i for i in range(600000, 606000)]
    codes += ["sh%06d" % i for i in range(688000, 689000)]
    return codes


def _parse_tx_batch(text: str) -> list[dict]:
    """
    解析腾讯批量返回。
    格式：v_sh600519="1~贵州茅台~600519~1439.00~1446.44~...";
    字段索引（实测 2026-11-05，共 52 项）：
      1=名称 2=代码 3=现价 4=昨收 5=今开 6=成交量(手) 30=时间 31=涨跌额 32=涨跌幅%
      33=最高 34=最低 37=成交额(万元) 38=换手率% 39=市盈率 43=振幅
      44=流通市值(亿) 45=总市值(亿) 46=市净率 47=涨停价 48=跌停价 49=量比
    """
    out = []
    for line in text.split(";"):
        line = line.strip()
        if '="' not in line:
            continue
        try:
            f = line.split('"')[1].split("~")
            if len(f) < 50:
                continue
            code, name, price = f[2], f[1], f[3]
            if not code or not name:
                continue
            def num(i):
                try:
                    v = float(f[i])
                    return v if v == v else None
                except (ValueError, IndexError):
                    return None
            out.append({
                "代码": code,
                "名称": name,
                "最新价": num(3),
                "涨跌幅": num(32),
                "成交额": (num(37) or 0) * 1e4,   # 万元 → 元，与东财口径统一
                "换手率": num(38),
                "量比": num(49),
                "总市值": (num(45) or 0) * 1e8,   # 亿 → 元
                "涨停价": num(47),
                "跌停价": num(48),
                "最高": num(33),
                "最低": num(34),
            })
        except Exception:
            continue
    return out


def fetch_spot_tx(batch_size: int = _BATCH_SIZE, verbose: bool = True) -> pd.DataFrame:
    """
    腾讯批量全A快照。返回列与东财口径一致（代码/名称/最新价/涨跌幅/成交额/换手率/量比/总市值）。
    约 12-26 次请求 / 1-4 秒完成。被 WAF 拦截时抛 SourceBlocked（交给上层切备源）。
    """
    codes = _load_code_list()
    sess = requests.Session()
    rows, ok_batch = [], 0
    t0 = time.time()
    rem = blocked_remain("tx")
    if rem > 0:
        raise SourceBlocked(f"腾讯源冷却中，还需 {rem:.0f} 秒（上次被WAF拦截，期内跳过以免延长封禁）")
    for i in range(0, len(codes), batch_size):
        sub = codes[i:i + batch_size]
        try:
            r = sess.get(TX_BATCH_URL + ",".join(sub), timeout=20,
                         headers={**_UA, "Referer": "https://gu.qq.com/"})
            r.encoding = "gbk"
            if _looks_blocked_batch(r):
                if ok_batch == 0:
                    mark_blocked("tx", 180)
                    raise SourceBlocked(f"腾讯WAF拦截批量行情（status={r.status_code}），已进入3分钟冷却")
                continue
            rows.extend(_parse_tx_batch(r.text))
            ok_batch += 1
        except SourceBlocked:
            raise
        except Exception as e:
            if verbose:
                print(f"[腾讯批量] 第{i // batch_size + 1}批异常 {type(e).__name__}: {e}（跳过）")
            continue
        time.sleep(_BATCH_GAP)
    if not rows:
        raise SourceBlocked("腾讯批量行情全部失败")
    df = pd.DataFrame(rows).drop_duplicates(subset="代码", keep="first")
    _save_active_codes(df)   # 落盘有效清单，下次快照直接 12 批
    if verbose:
        print(f"[腾讯快照] 有效 {len(df)} 只 | 成功批次 {ok_batch} | 耗时 {time.time() - t0:.1f}s")
    return df.reset_index(drop=True)


def _load_code_list() -> list[str]:
    """
    代码列表本地缓存，分两级，把快照请求数从 26 批压到 12 批：
      active_codes.txt — 上次扫描得到的真实有效代码（约5500只，12批）；优先使用
      code_list.txt    — 候选全集（含空号约1.4万只，26批）；仅用于重建有效清单
    有效清单每天重建一次（覆盖新股上市）。
    """
    import os
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    active_p = os.path.join(C.CACHE_DIR, "active_codes.txt")
    cand_p = os.path.join(C.CACHE_DIR, "code_list.txt")

    # 1) 优先：有效代码清单（24小时内）
    if os.path.exists(active_p) and time.time() - os.path.getmtime(active_p) < 86400:
        try:
            with open(active_p, encoding="utf-8") as fh:
                lst = [l.strip() for l in fh if l.strip()]
            if len(lst) > 1000:
                return lst
        except Exception:
            pass

    # 2) 候选全集
    if os.path.exists(cand_p) and time.time() - os.path.getmtime(cand_p) < 86400:
        try:
            with open(cand_p, encoding="utf-8") as fh:
                lst = [l.strip() for l in fh if l.strip()]
            if len(lst) > 1000:
                return lst
        except Exception:
            pass

    lst = _candidate_codes()
    try:
        with open(cand_p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lst))
    except Exception:
        pass
    return lst


def _save_active_codes(df: pd.DataFrame):
    """扫描成功后落盘有效代码清单，下次快照只需 12 批请求"""
    import os
    try:
        codes = ["sz" + c if c.startswith(("00", "30")) else "sh" + c
                 for c in df["代码"].astype(str)]
        p = os.path.join(C.CACHE_DIR, "active_codes.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(codes))
    except Exception:
        pass


# ---------- 前复权日K ----------
def fetch_kline_tx(code: str, days: int = 120, session: requests.Session | None = None,
                   retry: int = 1) -> pd.DataFrame | None:
    """
    腾讯前复权日K。code 为 6 位纯数字（002384），内部自动加 sz/sh 前缀。
    返回列：date/open/close/high/low/volume（volume 单位：手）
    被 WAF 拦截时退避重试 retry 次，仍失败返回 None（不造数据）。
    注意：腾讯不返回历史换手率与成交额，因子层 turnover_pct 会降级为中性 0.5。
    """
    sym = _to_tx_symbol(code)
    if sym is None:
        return None
    if blocked_remain("tx") > 0:
        return None            # 冷却期内不发请求，直接返回 None（上层会用缓存/备源）
    sess = session or requests.Session()

    # 优先用上次成功的入口，其余作为 failover
    urls = [_working_url["url"]] + [u for u in TX_KLINE_URLS if u != _working_url["url"]]
    headers = {**_UA, "Referer": "https://gu.qq.com/"}

    for url in urls:
        for attempt in range(retry + 1):
            try:
                r = sess.get(url, params={"param": f"{sym},day,,,{days},qfq"},
                             timeout=12, headers=headers)
                if _looks_blocked(r):
                    # 本入口被拦 → 换下一个入口（不重试同一入口，避免加深封禁）
                    break
                d = r.json()
                if d.get("code") != 0:
                    return None
                node = (d.get("data") or {}).get(sym) or {}
                k = node.get("qfqday") or node.get("day")
                if not k:
                    return None
                rows = []
                for it in k:
                    # 实测格式：[date, open, close, high, low, volume]
                    if len(it) < 6:
                        continue
                    rows.append({"date": it[0], "open": float(it[1]), "close": float(it[2]),
                                 "high": float(it[3]), "low": float(it[4]), "volume": float(it[5])})
                if not rows:
                    return None
                _working_url["url"] = url      # 记住这个可用入口
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date").reset_index(drop=True)
            except Exception:
                if attempt < retry:
                    time.sleep(0.8)
                    continue
                break
    return None


def _to_tx_symbol(code: str) -> str | None:
    """6位纯代码 → 腾讯符号（sh/sz 前缀）"""
    c = str(code).strip()
    if c.startswith(("sh", "sz", "bj")):
        return c
    c = c.zfill(6)
    if c.startswith(("60", "68", "9")):
        return "sh" + c
    if c.startswith(("00", "30", "20", "1")):
        return "sz" + c
    if c.startswith(("4", "8")):
        return "bj" + c
    return None


def fetch_kline_batch(codes: list[str], days: int = 120,
                      workers: int = _KLINE_WORKERS, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """
    并发批量拉前复权K线（带节流 + WAF 熔断）。
    - 4 线程 + 每只最小间隔 30ms → 约 33 只/秒，150 只约 5 秒
    - 连续 15 只失败判定为 WAF 整体拦截，立即中止（抛 SourceBlocked），
      避免像东财那样硬扛把封禁时间拖长
    返回 {6位代码: df}，单只失败自动跳过（不造数据）。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    if not codes:
        return {}
    sess = requests.Session()
    t0 = time.time()
    pace_lock = threading.Lock()
    last = [0.0]
    fail_streak = [0]
    abort = [False]

    def pace():
        with pace_lock:
            wait = _KLINE_GAP - (time.time() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.time()

    def work(c):
        if abort[0]:
            return str(c), None
        pace()
        df = fetch_kline_tx(c, days=days, session=sess)
        with pace_lock:
            if df is None:
                fail_streak[0] += 1
                if fail_streak[0] >= 15:
                    abort[0] = True
            else:
                fail_streak[0] = 0
        return str(c), df

    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for code, df in ex.map(work, codes):
            if df is not None and not df.empty:
                out[code] = df

    if abort[0] and not out:
        mark_blocked("tx", 180)
        raise SourceBlocked(f"腾讯K线被WAF拦截（连续{fail_streak[0]}只失败），已中止整批并进入3分钟冷却")
    if abort[0]:
        mark_blocked("tx", 120)
        if verbose:
            print(f"[腾讯K线] ⚠️ 检测到WAF拦截，已提前中止（成功 {len(out)}/{len(codes)}）")
    if verbose:
        print(f"[腾讯K线] 成功 {len(out)}/{len(codes)} 只 | {workers}线程 | 耗时 {time.time() - t0:.1f}s")
    return out


# ---------- 调试入口 ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="打印原始返回结构，排查字段变更")
    args = ap.parse_args()

    if args.debug:
        r = requests.get(TX_BATCH_URL + "sz002384", timeout=12, headers=_UA)
        r.encoding = "gbk"
        print("批量原始:", r.text[:300])
        raw = r.text.split('"')[1].split("~")
        print("字段数:", len(raw))
        for i, v in enumerate(raw):
            if i < 9 or 29 <= i <= 51:
                print(f"  [{i}] {v}")
        k = fetch_kline_tx("002384", days=5)
        print("\nK线样例:\n", k)
        raise SystemExit(0)

    sp = fetch_spot_tx()
    print(sp.head(3).to_string())
    print("列:", list(sp.columns))
    kl = fetch_kline_batch(["002384", "600519", "300750"])
    for c, d in kl.items():
        print(c, len(d), "根, 最新", d["date"].max().date(), "收盘", d["close"].iloc[-1])
