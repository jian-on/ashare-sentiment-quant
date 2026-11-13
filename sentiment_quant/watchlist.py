# -*- coding: utf-8 -*-
"""
watchlist.py — 模块四B：自选意向股全面分析（实时数据驱动）

改动点：
- 输入自选股列表，实时快照(行情) + 前复权K线(因子) + 舆情LLM打分(新闻/股吧)
  + 市场情绪温度计(涨跌家数/涨停炸板) 四路数据融合
- 输出：单股综合卡片 + 全池因子排名，可存JSON供Streamlit面板读取

用法：
  python watchlist.py --codes 002384,600519,300059
  （不传 --codes 则使用 config.WATCHLIST）
"""
import argparse
import json
import os
from datetime import datetime

import pandas as pd

import config as C
import data_loader as dl
import factors as F
import market_sentiment as MS
from sentiment import score_stock_sentiment

# 自选意向股（可在此维护，也可运行时 --codes 覆盖）
WATCHLIST = ["002384", "600519", "300750", "002594"]


def analyze_watchlist(codes: list[str], with_sentiment: bool = True) -> dict:
    print(f"[自选分析] 标的: {codes}，开始拉取实时快照...")
    spot = dl.get_spot()

    # ---- 市场情绪温度计（一次）----
    thermo = MS.market_thermometer(spot=spot)   # 与打分同一份快照，避免截面时点错配
    print(f"[市场温度] market_score={thermo['market_score']} 涨停={thermo['detail'].get('涨停家数')} "
          f"炸板率={thermo['detail'].get('炸板率')}")

    # ---- 逐股因子（单只失败不影响其他标的）----
    pool = {}
    blocked = []
    for c in codes:
        try:
            h = dl.fetch_hist(c)
        except dl.RateLimited:
            blocked.append(c)          # 数据源被拦截：整批记录，不再逐只重试（避免延长封禁）
            break
        if h is not None:
            pool[c] = h
    if blocked:
        print(f"[警告] 数据源被临时拦截，已中止后续请求（未完成标的: {blocked + [x for x in codes if x not in pool and x not in blocked]}）")
    missing = [c for c in codes if c not in pool]
    if missing and not blocked:
        print(f"[警告] 无法获取K线(退市/代码错/停牌?): {missing}")

    # ---- 舆情打分 ----
    sent_map = {}
    sentiment_detail = {}
    if with_sentiment:
        for c in codes:
            r = score_stock_sentiment(c)
            sent_map[c] = r.get("mean_sentiment", 0.0)
            sentiment_detail[c] = {k: v for k, v in r.items() if k != "posts"}
            print(f"[舆情] {c}: mean={sent_map[c]} source={r.get('source')} n={r.get('n_posts')}")

    # ---- 因子打分（自选池内横截面）----
    scored = F.score_pool(pool, spot, sentiment_map=sent_map,
                          market_score=thermo["market_score"]) if pool else pd.DataFrame()

    # ---- 实时行情卡片 ----
    cards = []
    spot_idx = spot.set_index(spot["代码"].astype(str))
    for c in codes:
        if c not in spot_idx.index:
            continue
        row = spot_idx.loc[c]
        cards.append({"code": c, "名称": str(row["名称"]),
                      "现价": row.get("最新价"), "涨跌幅": row.get("涨跌幅"),
                      "成交额亿": round(float(row.get("成交额") or 0) / 1e8, 2),
                      "换手率": row.get("换手率"), "量比": row.get("量比")})

    report = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "market": thermo, "cards": cards,
              "ranking": scored.to_dict("records") if not scored.empty else [],
              "sentiment_detail": sentiment_detail}
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    out = os.path.join(C.CACHE_DIR, "watchlist_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"[输出] 报告已存 {out}")
    return report


def print_report(report: dict):
    print("\n" + "=" * 70)
    print(f"自选股全面分析  @{report['generated_at']}   风险提示：仅学习验证，禁止实盘重仓")
    print("=" * 70)
    d = report["market"]["detail"]
    print(f"市场温度 {report['market']['market_score']:+.2f} | 上涨{d.get('上涨')} 下跌{d.get('下跌')} "
          f"涨停{d.get('涨停家数')} 炸板率{d.get('炸板率')}")
    print("\n-- 实时行情卡片 --")
    print(pd.DataFrame(report["cards"]).to_string(index=False))
    if report["ranking"]:
        rk = pd.DataFrame(report["ranking"])
        cols = [c for c in ["code", "名称", "composite", "sentiment", "reversal_5d",
                            "turnover_z", "vol_trend", "pos_in_20d"] if c in rk.columns]
        print("\n-- 综合打分排名(高分在前) --")
        print(rk[cols].round(3).to_string(index=False))
    print("\n-- 舆情摘要 --")
    for c, s in report["sentiment_detail"].items():
        print(f"{c}: 情绪均值={s.get('mean_sentiment')} 噪声占比={s.get('noise_ratio')} "
              f"题材={s.get('top_themes')} 来源={s.get('source')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=None, help="逗号分隔，覆盖默认自选池")
    ap.add_argument("--no-sentiment", action="store_true", help="跳过舆情(快)")
    a = ap.parse_args()
    codes = a.codes.split(",") if a.codes else WATCHLIST
    print_report(analyze_watchlist(codes, with_sentiment=not a.no_sentiment))
