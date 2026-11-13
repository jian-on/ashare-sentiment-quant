# -*- coding: utf-8 -*-
"""
sentiment.py — 模块二：舆情情感打分（对接通义千问/Qwen，规则2）

改动点：
- 主数据源：东财个股新闻 search-api（2026-11实测可用）
- 股吧 gbapi 接口当前反爬加固返回空列表（re=[]），保留占位，见文末调试方案
- akshare 1.20.22 的 stock_news_em 与 pandas 3.0 不兼容（\\u3000正则在pyarrow报错），
  故新闻走 requests 直连上游，绕开 akshare 封装层
- LLM：通义千问 OpenAI 兼容接口（dashscope compatible-mode），
  环境变量 DASHSCOPE_API_KEY 提供密钥；无密钥时自动降级为【本地规则词典】，
  降级结果显式打上 source="RULE_BASED" 标签（规则7：绝不把规则分冒充LLM分）
- 输出标准JSON（规则2）：sentiment(-1~1) / theme / is_noise

风险点：
- 新闻偏媒体口径（理性），股吧偏散户情绪（极端），两者情绪分布不同，
  混用时需在回测中分开验证（本模块 source 字段已区分）
- LLM 打分存在漂移：同一文本不同批次分数可能不一致（temperature=0.1缓解，不能根除）
"""
import os
import re
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import akshare as ak

QWEN_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-plus"          # 可换 qwen-turbo(快/便宜) 或 qwen-max(强/贵)
BATCH_SIZE = 15                    # 每次API调用打包的帖子数（省token）
MAX_POSTS = 50                     # 每只股票最多拉取的帖子数

# ---------- Prompt（规则2：输出标准JSON） ----------
SYSTEM_PROMPT = """你是A股财经舆情分析师。对每条股吧帖子做三项判断，严格输出JSON数组，不要任何多余文字：
1. sentiment: 情绪分，-1到1的浮点数。-1=极度看空恐慌，0=中性/纯提问，1=极度看多狂热。
   注意：A股散户语境中"笑嘻嘻""关灯吃面"等反讽要按实际含义判断；只陈述事实不给观点按0处理。
2. theme: 题材标签字符串，从帖子内容提炼，如"机器人""固态电池""算力""重组""业绩""无明确题材"。
3. is_noise: 布尔值。以下视为灌水帖(填true)：纯表情、无意义喊单（"冲""干"）、广告、
   引战对骂、与个股基本面/消息面无关的水帖。
输出格式严格为：[{"i":0,"sentiment":0.5,"theme":"xx","is_noise":false}, ...]，i为输入序号。"""

# ---------- 本地规则词典（降级方案，仅用于接口调试，不用于正式回测结论） ----------
BULL_WORDS = ["涨停", "连板", "起飞", "爆发", "利好", "主线", "龙头", "新高", "突破", "看多", "加仓", "满仓"]
BEAR_WORDS = ["跌停", "暴跌", "崩", "利空", "出货", "割肉", "逃命", "看空", "清仓", "垃圾", "爆雷", "踩雷"]
NOISE_PAT = re.compile(r"^(冲|干|顶|哈哈|\?|？|！|!)+[!！?？。.]*$")

def rule_based_score(text: str) -> dict:
    """规则词典打分（降级用）。返回与LLM相同结构的JSON。"""
    t = str(text)
    bull = sum(t.count(w) for w in BULL_WORDS)
    bear = sum(t.count(w) for w in BEAR_WORDS)
    if bull + bear == 0:
        s = 0.0
    else:
        s = round((bull - bear) / (bull + bear + 1), 2)  # 除数+1压缩极端值
    return {"sentiment": max(-1, min(1, s)), "theme": "无明确题材", "is_noise": bool(NOISE_PAT.match(t.strip()))}


# ---------- 数据抓取 ----------
NEWS_API = "https://search-api-web.eastmoney.com/search/jsonp"
GUBA_API = "https://gbapi.eastmoney.com/webarticlelist/api/Article/Articlelist"


def _ex_prefix(code: str) -> str:
    """000/001/002/003->sz，60->sh，68->sh（科创板），4/8->bj"""
    if code.startswith(("60", "68")):
        return "sh" + code
    if code.startswith(("0", "3")):
        return "sz" + code
    return "bj" + code


def fetch_news_posts(code: str, max_posts: int = MAX_POSTS) -> pd.DataFrame | None:
    """
    主数据源：东财个股新闻（requests直连，绕开akshare的pandas3.0兼容bug）。
    2026-11-02实测：字段 date/image/code/title/content/mediaName/url。
    """
    param = json.dumps({"uid": "", "keyword": code, "type": ["cmsArticleWebOld"],
                        "client": "web", "clientType": "web", "clientVersion": "curr",
                        "param": {"cmsArticleWebOld": {"searchScope": "default",
                                   "sort": "default", "pageIndex": 1,
                                   "pageSize": min(max_posts, 100),
                                   "preTag": "", "postTag": ""}}})
    try:
        r = requests.get(NEWS_API, params={"cb": "jq", "param": param}, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        body = r.text.strip()
        if body.startswith("jq("):
            body = body[3:-1]
        items = json.loads(body)["result"]["cmsArticleWebOld"]
        if not items:
            return None
        df = pd.DataFrame(items)
        df["text"] = (df["title"].fillna("").astype(str) + " "
                      + df["content"].fillna("").astype(str)).str.strip()
        df = df[df["text"].str.len() > 10].head(max_posts)
        return df[["text", "date"]].rename(columns={"date": "time"}).reset_index(drop=True)
    except Exception as e:
        print(f"[新闻抓取失败 {code}] {type(e).__name__}: {e}")
        return None


def fetch_guba_posts(code: str, max_posts: int = MAX_POSTS) -> pd.DataFrame | None:
    """
    股吧帖子（gbapi直连）。2026-11-02实测：无论code带不带交易所前缀均返回 re=[]，
    疑似需要登录态cookie或签名。保留占位，主流程自动落到新闻源。
    调试方案见文件末尾注释。
    """
    try:
        r = requests.get(GUBA_API,
                         params={"code": _ex_prefix(code), "type": 0, "index": 1, "count": max_posts},
                         timeout=15, headers={"User-Agent": "Mozilla/5.0",
                                              "Referer": f"https://guba.eastmoney.com/list,{code}.html"})
        d = r.json()
        items = d.get("re") or []
        if not items:
            return None
        df = pd.DataFrame(items)
        text = df["post_title"].fillna("").astype(str) + " " + df.get("post_content", pd.Series([""] * len(df))).fillna("").astype(str)
        df["text"] = text.str.strip()
        df = df[df["text"].str.len() > 0].head(max_posts)
        return df[["text"]].assign(time=df.get("post_display_time")).reset_index(drop=True)
    except Exception as e:
        print(f"[股吧抓取失败 {code}] {type(e).__name__}: {e}")
        return None


# ---------- LLM 批量打分 ----------
def qwen_batch_score(texts: list[str]) -> list[dict] | None:
    """调用通义千问，一次打包BATCH_SIZE条。失败返回None（不造数据）。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        return None
    user_msg = "\n".join(f"{{i:{i}, text: {t[:200]}}}" for i, t in enumerate(texts))
    try:
        resp = requests.post(QWEN_BASE, timeout=60, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": QWEN_MODEL, "temperature": 0.1,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": user_msg}]})
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # 提取JSON数组（LLM偶尔会包```json围栏）
        m = re.search(r"\[.*\]", content, re.S)
        if not m:
            print(f"[LLM输出解析失败] 原文: {content[:200]}")
            return None
        items = json.loads(m.group())
        out = [None] * len(texts)
        for it in items:
            i = int(it.get("i", -1))
            if 0 <= i < len(texts):
                out[i] = {"sentiment": float(max(-1, min(1, it["sentiment"]))),
                          "theme": str(it.get("theme", "无明确题材")),
                          "is_noise": bool(it.get("is_noise", False))}
        return out
    except Exception as e:
        print(f"[Qwen API失败] {type(e).__name__}: {e}")
        return None


def score_stock_sentiment(code: str, use_news_fallback: bool = True) -> dict:
    """
    单只股票完整舆情打分流水线。
    返回：{"code", "source", "n_posts", "n_used", "mean_sentiment",
           "noise_ratio", "top_themes", "posts":[{标准JSON}]}
    """
    posts = fetch_guba_posts(code)
    src = "GUBA"
    if (posts is None or posts.empty) and use_news_fallback:
        posts = fetch_news_posts(code)
        src = "NEWS"
    if posts is None or posts.empty:
        return {"code": code, "source": src, "n_posts": 0, "error": "无可用舆情文本"}

    texts = posts["text"].tolist()
    n_api_fail = 0
    if os.environ.get("DASHSCOPE_API_KEY"):
        results, n_api_fail = [], 0
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            scored = qwen_batch_score(batch)
            if scored is None:
                n_api_fail += len(batch)
                scored = [rule_based_score(t) for t in batch]
                results.extend(scored)
            else:
                results.extend(scored)
            time.sleep(0.3)
        source = "QWEN" if n_api_fail == 0 else f"QWEN+RULE({n_api_fail}条降级)"
    else:
        results = [rule_based_score(t) for t in texts]
        source = "RULE_BASED"

    df = pd.concat([posts.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    valid = df[~df["is_noise"]]
    themes = valid["theme"].value_counts().head(3).to_dict() if len(valid) else {}

    return {"code": code, "source": source, "n_posts": len(df),
            "n_used": len(valid), "noise_ratio": round(1 - len(valid) / max(len(df), 1), 2),
            "mean_sentiment": round(valid["sentiment"].mean(), 3) if len(valid) else 0.0,
            "top_themes": themes,
            "posts": df[["text", "sentiment", "theme", "is_noise"]].to_dict("records")}


def batch_score(codes: list[str], workers: int = 4) -> list[dict]:
    """并发打分（控制并发，防反爬/限流）。"""
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_stock_sentiment, c): c for c in codes}
        for f in as_completed(futs):
            r = f.result()
            print(f"[舆情] {r['code']}: source={r.get('source')} "
                  f"情绪均值={r.get('mean_sentiment', '-')} 帖子={r.get('n_posts')}")
            out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="002384,600519", help="逗号分隔的股票代码")
    ap.add_argument("--debug", action="store_true", help="打印数据源连通性")
    args = ap.parse_args()
    if args.debug:
        print("== 新闻源（主）==")
        n = fetch_news_posts("002384", max_posts=5)
        print(None if n is None else n.to_string())
        print("== 股吧源（占位，预期返回None）==")
        g = fetch_guba_posts("002384", max_posts=5)
        print("guba:", "None(空列表/反爬)" if g is None else g.head())
        raise SystemExit(0)
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("!! 未检测到 DASHSCOPE_API_KEY，将使用 RULE_BASED 规则词典降级打分（仅调试用）\n"
              "!! 正式使用请: export DASHSCOPE_API_KEY=sk-xxxx")
    report = batch_score(args.codes.split(","))
    print(json.dumps([{k: v for k, v in r.items() if k != "posts"} for r in report],
                     ensure_ascii=False, indent=2))

# ========== 股吧接口调试方案（规则7备选路径，按优先级排序） ==========
# 1. 浏览器打开 https://guba.eastmoney.com/list,002384.html，F12抓包 Network 里
#    真实的帖子列表请求（注意是否携带 cookie / 签名参数），把 headers 补进 fetch_guba_posts
# 2. 用 selenium/playwright 渲染页面后解析 DOM（慢但稳，50只股票池约需10分钟）
# 3. 备选文本源：雪球个股讨论（需cookie）、新浪财经股吧、同花顺快讯
#    ak.stock_zh_a_alerts_cls / ak.stock_info_global_cls（财联社电报，市场级情绪）
# 4. 本模型当前用个股新闻做主源，媒体口径偏理性，散户极端情绪信号偏弱——
#    这是已知偏差，回测时会单独对比 GUBA/NEWS 两种 source 的因子有效性
