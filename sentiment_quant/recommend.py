# -*- coding: utf-8 -*-
"""
recommend.py — 市场分析驱动的「横截面因子相对排序观测」清单

2026-11-13 多智能体审计后的修订，详见 RESEARCH_2026-11-13.md：
  1. 修复选择偏差：原来先按「无舆情分」取 Top10，再只对这 10 只补舆情重排，
     导致权重最高的舆情因子永远无法影响「谁入选」。现在先取确定性候选集
     （默认 60 只 ≥ 5×top_n），集内全量补舆情，再在同一横截面里一次成榜。
  2. 舆情分做横截面标准化后才进 composite（在 factors.score_pool 内完成），
     不再把 [-1,1] 的原始分与 z 分数直接混加。
  3. 温度计复用同一份 spot，避免 1 小时缓存快照与 60 秒打分快照错配。
  4. 「参考理由」只从实际生效的权重里取贡献，排除常数/零方差因子。
  5. 新增数据时间、有效权重、舆情覆盖率、市场状态等诊断，降级不再静默。

⚠️ 这是因子得分排序的观测清单，不是荐股；模型只做横截面相对比较，
   无任何涨跌预测能力，未经样本外回测，禁止实盘重仓。
"""
import datetime as dt
import json
import os
import time

import pandas as pd

import config as C
import data_loader as dl
import factors as F
import market_sentiment as MS
import regime as RG
from sentiment import batch_score, score_stock_sentiment

FACTOR_LABELS = {
    "reversal_5d": "5日超跌反转", "pos_in_20d": "20日低位", "low_vol": "低波动",
    "vol_trend": "量能趋势健康", "turnover_z": "换手活跃", "volume_ratio": "量比放大",
    "sentiment": "舆情偏暖", "market_beta": "市场情绪加成",
}

_SENT_CACHE = os.path.join(C.CACHE_DIR, "sentiment_snapshot.json")


def _load_sent_cache() -> dict:
    try:
        with open(_SENT_CACHE, "r", encoding="utf-8") as f:
            d = json.load(f)
        if (time.time() - d.get("ts", 0)) <= C.SENTIMENT_CACHE_TTL_SEC \
                and d.get("date") == dt.date.today().isoformat():
            return d.get("data", {})
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def _save_sent_cache(data: dict) -> None:
    try:
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        with open(_SENT_CACHE, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "date": dt.date.today().isoformat(),
                       "data": data}, f, ensure_ascii=False)
    except OSError:
        pass


def _collect_sentiment(codes: list[str]) -> tuple[dict, dict]:
    """
    对候选集全量补舆情（带当日缓存）。失败的单只记中性 0 并在 detail 里标注，
    绝不把「没取到」当成「舆情差」而静默惩罚。
    """
    cache = _load_sent_cache()
    need = [c for c in codes if c not in cache]
    detail = {c: cache[c] for c in codes if c in cache}
    if need:
        try:
            for r in batch_score(need, workers=4):
                c = str(r.get("code"))
                item = {k: v for k, v in r.items() if k != "posts"}
                if "error" in r:
                    item["mean_sentiment"] = 0.0
                    item["degraded"] = True
                detail[c] = item
                cache[c] = item
        except Exception as e:
            print(f"[舆情] 批量打分失败，全部记中性：{type(e).__name__}: {e}")
            for c in need:
                detail[c] = {"code": c, "mean_sentiment": 0.0, "degraded": True}
                cache[c] = detail[c]
        _save_sent_cache(cache)
    sent_map = {c: float(v.get("mean_sentiment", 0.0) or 0.0) for c, v in detail.items()}
    return sent_map, detail


def _reason(row: pd.Series, weights: dict, top_k: int = 2) -> str:
    """取加权贡献最大的 k 个生效因子；只从有效权重里选，排除常数/零方差因子"""
    contrib = {}
    for k, w in weights.items():
        if k in row and pd.notna(row[k]) and k != "market_beta":
            contrib[k] = float(row[k]) * w
    if not contrib:
        return "无有效因子"
    top = sorted(contrib.items(), key=lambda x: -abs(x[1]))[:top_k]
    return "；".join(f"{FACTOR_LABELS.get(k, k)}({v:+.2f})" for k, v in top)


def build_recommend_list(top_n: int = 10, pool_size: int = 300,
                         with_sentiment: bool = True) -> dict:
    asof = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    spot = dl.get_spot()
    uni = dl.filter_universe(spot)
    pool = dl.build_stock_pool(uni, max_pool=pool_size)

    # 温度计与打分复用同一份快照，避免截面时点错配
    thermo = MS.market_thermometer(spot=spot)
    weights, state, regime_note = RG.resolve_weights(thermo.get("detail", {}))
    RG.save_state(state, thermo.get("market_score"))

    # ---- 一期：技术面排序，取确定性候选集（不再是最终 Top10）----
    pre = F.score_pool(pool, spot, market_score=thermo["market_score"], weights=weights)
    coverage_n = min(len(pre), C.SENTIMENT_COVERAGE_N if C.SENTIMENT_COVERAGE_N > 0 else len(pre))
    coverage_n = max(coverage_n, top_n)
    cand_codes = [str(c) for c in pre.head(coverage_n)["code"]]

    sent_map, sent_detail = {}, {}
    if with_sentiment:
        sent_map, sent_detail = _collect_sentiment(cand_codes)

    # ---- 二期：只在同一候选集横截面内打分，未打分者不进入，避免「0 分=看空」误判 ----
    cand_pool = {c: pool[c] for c in cand_codes if c in pool}
    scored, diag = F.score_pool(cand_pool, spot, sentiment_map=sent_map or None,
                                market_score=thermo["market_score"], weights=weights,
                                return_diag=True)
    top = scored.head(top_n).copy()
    eff_w = diag["weights_effective"]
    top["参考理由"] = top.apply(lambda r: _reason(r, eff_w), axis=1)

    spot_idx = spot.set_index(spot["代码"].astype(str))
    rows = []
    for _, r in top.iterrows():
        c = r["code"]
        if c in spot_idx.index:
            s = spot_idx.loc[c]
            rows.append({
                "代码": c, "名称": r["名称"], "现价": s.get("最新价"),
                "涨跌幅%": s.get("涨跌幅"), "成交额(亿)": round(float(s.get("成交额") or 0) / 1e8, 2),
                "综合分": round(float(r["composite"]), 3),
                "舆情分": round(float(sent_map.get(c, 0.0)), 3),
                "参考理由": r["参考理由"],
            })
    df = pd.DataFrame(rows)
    n_degraded = sum(1 for v in sent_detail.values() if v.get("degraded"))
    return {
        "market": thermo,
        "list": df,
        "sentiment_detail": sent_detail,
        "n_pool": len(pool),
        "n_universe": len(uni),
        # ---- 新增诊断 ----
        "asof": asof,
        "diag": diag,
        "state": state,
        "regime_note": regime_note,
        "coverage_n": coverage_n,
        "sentiment_degraded": n_degraded,
        "disclaimer": "横截面因子相对排序观测，非涨跌预测、非荐股；未经样本外回测验证",
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--no-sentiment", action="store_true")
    a = ap.parse_args()
    r = build_recommend_list(top_n=a.top, pool_size=a.pool,
                             with_sentiment=not a.no_sentiment)
    print(f"\n数据时间 {r['asof']} | 状态 {r['state']} | {r['regime_note']}")
    print(f"有效权重合计 {r['diag']['effective_weight_total']} | "
          f"剔除因子 {r['diag']['dropped_factors']} | "
          f"舆情覆盖率 {r['diag']['sentiment_coverage']} | 降级 {r['sentiment_degraded']} 只")
    print(r["list"].to_string(index=False))
