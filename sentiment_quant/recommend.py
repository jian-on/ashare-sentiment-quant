# -*- coding: utf-8 -*-
"""
recommend.py — 模块五：市场分析驱动的个股参考清单

改动点：
- 流程：实时快照 → 四重过滤 → 300只候选池 → 因子打分 → 市场温度计
  → 仅对Top候选补舆情打分（省时） → 重新加权 → 输出带"参考理由"的清单
- 输出每只股的参考理由 = 综合分中贡献最大的2个因子（可解释，非黑箱）

风险点（规则5）：
- 这是"因子得分排序的观测清单"，不是荐股；题材情绪因子拥挤度高、易失效
- 高分≠会涨：模型只做横截面相对比较，无任何涨跌预测能力
"""
import pandas as pd

import config as C
import data_loader as dl
import factors as F
import market_sentiment as MS
from sentiment import score_stock_sentiment

FACTOR_LABELS = {
    "reversal_5d": "5日超跌反转", "pos_in_20d": "20日低位", "low_vol": "低波动",
    "vol_trend": "量能趋势健康", "turnover_z": "换手活跃", "volume_ratio": "量比放大",
    "sentiment": "舆情偏暖", "market_beta": "市场情绪加成",
}


def _reason(row: pd.Series, top_k: int = 2) -> str:
    """取加权贡献最大的k个因子拼成理由串"""
    contrib = {}
    for k, w in C.FACTOR_WEIGHTS.items():
        if k in row and pd.notna(row[k]):
            contrib[k] = float(row[k]) * w
    top = sorted(contrib.items(), key=lambda x: -x[1])[:top_k]
    return "；".join(f"{FACTOR_LABELS.get(k, k)}({v:+.2f})" for k, v in top)


def build_recommend_list(top_n: int = 10, pool_size: int = 300,
                         with_sentiment: bool = True) -> dict:
    spot = dl.get_spot()
    uni = dl.filter_universe(spot)
    pool = dl.build_stock_pool(uni, max_pool=pool_size)
    thermo = MS.market_thermometer()

    scored = F.score_pool(pool, spot, market_score=thermo["market_score"])
    top = scored.head(top_n).copy()

    # 只对Top候选补舆情（300只全打太慢），打完重算综合分
    sent_detail = {}
    if with_sentiment:
        for c in top["code"]:
            r = score_stock_sentiment(c)
            top.loc[top["code"] == c, "sentiment"] = r.get("mean_sentiment", 0.0)
            sent_detail[c] = {k: v for k, v in r.items() if k != "posts"}
        weights = C.FACTOR_WEIGHTS
        top["composite"] = sum(top[k].astype(float) * w for k, w in weights.items() if k in top.columns)
        top = top.sort_values("composite", ascending=False).head(top_n)

    top["参考理由"] = top.apply(_reason, axis=1)

    # 合并实时行情
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
                "舆情分": round(float(r.get("sentiment") or 0), 3),
                "参考理由": r["参考理由"],
            })
    df = pd.DataFrame(rows)
    return {"market": thermo, "list": df, "sentiment_detail": sent_detail,
            "n_pool": len(pool), "n_universe": len(uni)}
