# -*- coding: utf-8 -*-
"""
market_sentiment.py — 模块四A：市场情绪温度计（实时，规则2的"市场情绪"主线）

改动点：
- 乐咕乐股涨跌家数 + 东财涨停/炸板/跌停池，全部实时接口（2026-11实测存在）
- 输出 -1~1 的 market_score，供 factors.score_pool 的 market_beta 因子直接使用

风险点：
- 情绪温度计反映的是"当下"，短线情绪周期（冰点→回暖→高潮→退潮）切换极快，
  收盘后与盘中读数含义不同；本模块不做择时建议，只输出观测值
"""
import datetime as dt

import numpy as np
import pandas as pd

import akshare as ak


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        print(f"[{fn.__name__} 失败] {type(e).__name__}: {e}")
        return None


def _score_from_counts(up, down, n_zt, n_zb) -> tuple[float, dict]:
    """由家数统计合成情绪分：涨跌比0.5 + 涨停强度0.3 + 炸板率惩罚0.2"""
    # 2026-11-13 修复：全 0 时原式会算出 -0.3 的「偏空」值并被上层当成真实读数。
    # 现在显式返回中性 0 并打降级标记，绝不静默伪装成有效观测。
    if (up + down) == 0 and (n_zt + n_zb) == 0:
        return 0.0, {"涨跌比": None, "涨停家数": n_zt, "炸板家数": n_zb,
                     "炸板率": None, "数据质量": "DEGRADED: 无有效家数，返回中性0（非真实读数）"}
    breadth = (up - down) / (up + down) if (up + down) > 0 else 0.0
    zt_strength = min(n_zt / 100.0, 1.0)                     # 100家涨停记满分
    ban_rate = n_zb / (n_zb + n_zt) if (n_zb + n_zt) > 0 else None
    ban_pen = ban_rate if ban_rate is not None else 0.5
    score = 0.5 * breadth + 0.3 * (zt_strength * 2 - 1) + 0.2 * (1 - 2 * ban_pen)
    return max(-1.0, min(1.0, round(score, 3))), {
        "涨跌比": round(breadth, 3), "涨停家数": n_zt, "炸板家数": n_zb, "炸板率": ban_rate,
    }


def _from_spot(spot: pd.DataFrame) -> dict | None:
    """
    用全A快照本地计算温度计 —— 零额外请求，且比东财涨停池更实时。
    涨停/跌停用「涨停价/跌停价」字段精确判定（腾讯快照提供），
    炸板 = 最高价触及涨停但现价未封住。
    """
    if spot is None or len(spot) < 1000:
        return None
    px = pd.to_numeric(spot.get("最新价"), errors="coerce")
    chg = pd.to_numeric(spot.get("涨跌幅"), errors="coerce")
    if px is None or chg is None:
        return None
    valid = px.notna() & chg.notna()
    up = int((chg[valid] > 0).sum())
    down = int((chg[valid] < 0).sum())
    flat = int((chg[valid] == 0).sum())
    if up + down == 0:
        return None

    if {"涨停价", "跌停价", "最高"}.issubset(spot.columns):
        ztp = pd.to_numeric(spot["涨停价"], errors="coerce")
        dtp = pd.to_numeric(spot["跌停价"], errors="coerce")
        high = pd.to_numeric(spot["最高"], errors="coerce")
        # 2026-11-13 修复：0.999/1.001 是比例容差，100元股容差 0.11 元、
        # 200元股 0.22 元，会把「差几分钱没封住」误判为涨停。改用 1 个最小
        # 报价单位（0.01 元）的绝对容差，与价格高低无关。
        is_zt = valid & (ztp > 0) & ((px - ztp).abs() <= 0.01)
        is_dt = valid & (dtp > 0) & ((px - dtp).abs() <= 0.01)
        touched = valid & (ztp > 0) & (high >= ztp - 0.01)
        n_zt, n_dt = int(is_zt.sum()), int(is_dt.sum())
        n_zb = int((touched & ~is_zt).sum())
        exact = True
    else:
        # 东财备源快照无涨停价字段 → 按板块/ST 分别取涨跌幅阈值。
        # 旧实现统一 9.8%：20cm 股涨 9.8%~20% 被误计为涨停，ST 5cm 涨停反而漏计。
        code = spot["代码"].astype(str)
        name = spot["名称"].astype(str) if "名称" in spot.columns else pd.Series("", index=spot.index)
        is_20cm = code.str.startswith(("30", "688")).to_numpy()
        is_st = name.str.contains("ST", na=False).to_numpy()
        thr = np.where(is_20cm, 19.8, np.where(is_st, 4.8, 9.8))
        v = valid.to_numpy()
        n_zt = int(((chg.to_numpy() >= thr) & v).sum())
        n_dt = int(((chg.to_numpy() <= -thr) & v).sum())
        n_zb = 0        # 近似路径无「最高价触板」信息，炸板不计（宁缺勿造）
        exact = False

    score, detail = _score_from_counts(up, down, n_zt, n_zb)
    detail.update({"上涨": up, "下跌": down, "平盘": flat, "跌停家数": n_dt,
                   "market_score": score,
                   "数据来源": "快照本地计算(涨停价±0.01精确判定)" if exact
                              else "快照本地计算(按板块10/20cm+ST 5cm近似, 炸板不计)",
                   "数据质量": "OK" if exact else "APPROX: 无涨停价字段，涨跌停为阈值近似"})
    return {"market_score": score, "detail": detail}


def market_thermometer(spot: pd.DataFrame | None = None) -> dict:
    """
    实时市场情绪温度计。返回 {market_score(-1~1), 细项dict}
    优先用全A快照本地计算（零请求）；快照不可用时才回退东财接口（4次请求）。
    """
    # ---- 主路径：快照本地计算，不发任何额外请求 ----
    from_cache = spot is None
    if from_cache:
        try:
            import data_loader as dl
            # 允许 1 小时内的快照做收盘后复盘，但必须显式标注，避免与 60s 打分快照
            # 被误认为同一截面（2026-11-13 审计：时点错配不可静默）
            spot = dl._read_cache("spot", ttl_sec=3600)
        except Exception:
            spot = None
    r = _from_spot(spot)
    if r is not None:
        if from_cache:
            r["detail"]["数据质量"] = r["detail"].get("数据质量", "OK") + \
                                    " | 使用最长1小时缓存快照（非打分快照同一时点）"
        return r

    # ---- 兜底：东财接口（仅在快照不可用时）----
    import sources as SRC
    detail = {}
    if SRC.blocked_remain("em") > 0:
        detail["说明"] = f"无可用快照，东财冷却中（还需{SRC.blocked_remain('em'):.0f}秒），返回中性值"
        detail["数据质量"] = "DEGRADED: 无有效数据，中性0非真实读数"
        return {"market_score": 0.0, "detail": detail}

    up, down = 0.0, 0.0
    act = _safe(ak.stock_market_activity_legu)
    if act is not None and not act.empty:
        kv = dict(zip(act["item"], act["value"]))
        detail.update({k: kv.get(k) for k in ["上涨", "下跌", "平盘", "涨停", "跌停"] if k in kv})
        up, down = float(kv.get("上涨", 0) or 0), float(kv.get("下跌", 0) or 0)

    today = dt.date.today().strftime("%Y%m%d")
    zt = _safe(ak.stock_zt_pool_em, date=today)          # 涨停池
    zb = _safe(ak.stock_zt_pool_zbgc_em, date=today)     # 炸板池（涨停打开）
    dtn = _safe(ak.stock_zt_pool_dtgc_em, date=today)    # 跌停池
    n_zt = 0 if zt is None else len(zt)
    n_zb = 0 if zb is None else len(zb)
    n_dt = 0 if dtn is None else len(dtn)

    score, extra = _score_from_counts(up, down, n_zt, n_zb)
    detail.update(extra)
    detail.update({"跌停家数": n_dt, "market_score": score, "数据来源": "东财接口"})
    if "数据质量" not in detail:
        detail["数据质量"] = "OK"
    return {"market_score": score, "detail": detail}


def cls_flash(head_limit: int = 10) -> pd.DataFrame | None:
    """财联社电报（市场级消息面，辅助人工判断题材方向，不直接进因子）"""
    df = _safe(ak.stock_info_global_cls, symbol="全部")
    if df is None or df.empty:
        return None
    return df.head(head_limit)


if __name__ == "__main__":
    import json
    r = market_thermometer()
    print(json.dumps(r, ensure_ascii=False, indent=2))
    flash = cls_flash(5)
    if flash is not None:
        print("\n财联社快讯样例：")
        print(flash.to_string(max_colwidth=60))
