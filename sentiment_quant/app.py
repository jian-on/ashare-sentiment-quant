# -*- coding: utf-8 -*-
"""
app.py — A股题材情绪多因子终端（单页面版，全部区块纵向整合）

启动： streamlit run app.py --server.port 8501
区块： ①市场温度计 ②今日个股参考清单 ③自选意向股全面分析 ④全市场因子扫描
"""
import os
import time
import pandas as pd
import streamlit as st

import config as C
import data_loader as dl
from data_loader import RateLimited
import factors as F
import market_sentiment as MS
import watchlist as WL
import recommend as RC
import sources as SRC


def _guard(fn, *a, **kw):
    """执行重IO操作：成功则进入冷却；两源均失败则强制冷却并给出倒计时"""
    try:
        out = fn(*a, **kw)
        st.session_state["cooldown_until"] = time.time() + C.UI_COOLDOWN_SEC
        st.session_state["last_run_ts"] = time.time()
        return out
    except RateLimited as e:
        st.session_state["cooldown_until"] = time.time() + C.RATE_LIMIT_COOLDOWN
        st.session_state["last_run_ts"] = time.time()
        st.error(f"⛔ **行情数据源全部不可用**，已锁定按钮 **{C.RATE_LIMIT_COOLDOWN // 60} 分钟**。\n\n"
                 "主源（腾讯批量行情）与备源（东财）本次都请求失败，通常是网络波动。\n\n"
                 f"技术信息：{e}")
        return None


def _cooldown_ready() -> bool:
    """点击前冷却校验：未到时间则拦截并显示剩余秒数"""
    now = time.time()
    until = st.session_state.get("cooldown_until", 0)
    if now < until:
        remain = int(until - now)
        st.warning(f"⏳ **还需等待 {remain // 60} 分 {remain % 60} 秒** 才能再次拉取数据。\n\n"
                   f"这是防手滑的 {C.UI_COOLDOWN_SEC} 秒冷却（数据已拉到，先看完结果再重新扫描）。")
        return False
    return True


def _stale_warn():
    """仅在 config 允许降级时才可能出现 → 出现即明示，不让用户误以为是实时数据"""
    if dl.LAST_SPOT_STALE:
        st.warning("⚠️ **数据非实时**：本次使用了过期行情缓存，当日涨跌幅/量比等因子失真。")


@st.cache_data(ttl=C.SPOT_TTL_SEC, show_spinner=False)
def _thermo_cached():
    """温度计缓存：避免每次按钮点击重跑都重复打涨跌家数等接口"""
    return MS.market_thermometer()

st.set_page_config(page_title="A股题材情绪多因子终端", layout="wide")

# ========== 侧栏 ==========
with st.sidebar:
    st.header("⚠️ 风控提示")
    st.error("本模型仅用于学习验证，**禁止实盘重仓**。\n\n"
             "题材/舆情类因子天然容易失效；参考清单是因子得分排序的观测结果，\n\n"
             "**不是荐股，不含任何涨跌预测**。")
    st.subheader("设置")
    codes_input = st.text_input("自选意向股（逗号分隔）", ",".join(WL.WATCHLIST))
    use_sent = st.checkbox("启用舆情打分（无Key时为规则词典降级）", value=True)
    pool_size = st.slider("全市场扫描候选池规模", 50, 300, 150, 50)
    api_state = "🟢 Qwen已配置" if os.environ.get("DASHSCOPE_API_KEY") else "🟡 未配置Key(规则词典降级)"
    st.caption(f"LLM状态: {api_state}   |   行情源: 腾讯批量(主) + 东财(备)")
    st.caption(f"数据口径：**严格实时**（快照{C.SPOT_TTL_SEC}秒/K线{C.HIST_TTL_SEC}秒），"
               "两源均失败时直接报错，不使用过期缓存")

    # ---- 点击冷却状态（常驻显示剩余等待时间）----
    st.subheader("⏱️ 拉取冷却")
    _now = time.time()
    _until = st.session_state.get("cooldown_until", 0)
    if _now < _until:
        _remain = int(_until - _now)
        st.warning(f"冷却中：还需 **{_remain // 60} 分 {_remain % 60} 秒**\n\n"
                   f"（防手滑设计，{C.UI_COOLDOWN_SEC}秒后可再次扫描）")
    else:
        st.success("✅ 可以拉取数据")

    # ---- 数据源健康状态（被封后自动冷却，期内不发请求）----
    st.subheader("🔌 数据源")
    _tx, _em = SRC.blocked_remain("tx"), SRC.blocked_remain("em")
    st.caption(f"主源 腾讯批量：{'🟢 正常' if _tx <= 0 else f'🔴 冷却 {_tx:.0f}s'}")
    st.caption(f"备源 东财：{'🟢 正常' if _em <= 0 else f'🔴 冷却 {_em:.0f}s'}")
    if _tx > 0 or _em > 0:
        st.caption("⚠️ 冷却期内请求会被直接跳过，以免延长封禁")
    _last = st.session_state.get("last_run_ts")
    if _last:
        st.caption(f"上次拉取时间：{time.strftime('%H:%M:%S', time.localtime(_last))}")

codes = [c.strip() for c in codes_input.split(",") if c.strip()]

# ---------- ① 市场情绪温度计 ----------
st.title("📈 A股题材情绪多因子终端")
st.markdown("### 🌡️ ① 市场情绪温度计")
c1, c2, c3, c4, c5 = st.columns(5)
thermo = _thermo_cached()
d = thermo["detail"]
c1.metric("市场情绪分(-1~1)", f"{thermo['market_score']:+.2f}")
c2.metric("上涨/下跌", f"{d.get('上涨','-')} / {d.get('下跌','-')}")
c3.metric("涨停家数", d.get("涨停家数", "-"))
c4.metric("炸板率", d.get("炸板率", "-"))
c5.metric("跌停家数", d.get("跌停家数", "-"))
st.caption("口径：涨跌比0.5 + 涨停强度0.3 + 炸板率惩罚0.2。仅观测，不做择时建议。"
           "市场分>0.5情绪偏热(追高风险)，<-0.5情绪冰点(反转因子失效率上升)。")
st.divider()

# ---------- ② 今日个股参考清单 ----------
st.markdown("### 📋 ② 今日个股参考清单（市场分析驱动）")
st.caption("生成逻辑：实时快照 → 四重过滤(ST/停牌/次新/低流动性) → 候选池因子打分 → "
           "市场温度计加成 → Top候选补舆情 → 带参考理由输出")
if st.button("🚀 生成今日参考清单（约5秒）", type="primary") and _cooldown_ready():
    with st.spinner("全市场过滤 → 因子计算 → 舆情打分中..."):
        rec = _guard(RC.build_recommend_list, top_n=10, pool_size=pool_size, with_sentiment=use_sent)
        if rec is not None:
            st.session_state["rec"] = rec
if "rec" in st.session_state:
    _stale_warn()
    r = st.session_state["rec"]
    st.success(f"候选池 {r['n_pool']} 只（过滤前 {r['n_universe']}）｜市场温度 {r['market']['market_score']:+.2f}")
    st.dataframe(r["list"], width="stretch", hide_index=True)
    st.info("⚠️ 以上为因子得分排序的观测清单，仅作研究参考，非荐股。高分≠会涨，"
            "执行时还需评估滑点与冲击成本。")
st.divider()

# ---------- ③ 自选意向股全面分析 ----------
st.markdown("### ⭐ ③ 自选意向股全面分析")
c_btn1, c_btn2 = st.columns(2)
if c_btn1.button("分析自选股（实时行情+因子+舆情）") and _cooldown_ready():
    with st.spinner("拉取实时数据与舆情打分中..."):
        rep = _guard(WL.analyze_watchlist, codes, with_sentiment=use_sent)
        if rep is not None:
            st.session_state["report"] = rep
if c_btn2.button("清除自选股分析结果"):
    st.session_state.pop("report", None)
if "report" in st.session_state:
    _stale_warn()
    rep = st.session_state["report"]
    if rep.get("cards"):
        st.markdown("##### 实时行情卡片")
        st.dataframe(pd.DataFrame(rep["cards"]), width="stretch", hide_index=True)
    if rep.get("ranking"):
        st.markdown("##### 自选池内综合打分排名")
        st.dataframe(pd.DataFrame(rep["ranking"]).round(3), width="stretch", hide_index=True)
    st.markdown("##### 舆情摘要（来源已标注）")
    for c, s in rep.get("sentiment_detail", {}).items():
        st.write(f"**{c}** 情绪均值 {s.get('mean_sentiment')} | 噪声占比 {s.get('noise_ratio')} "
                 f"| 题材 {s.get('top_themes')} | 来源 {s.get('source')}")
st.divider()

# ---------- ④ 全市场因子扫描 ----------
st.markdown("### 📊 ④ 全市场因子扫描 Top20")
if st.button("运行全市场扫描（不补舆情，约5秒）") and _cooldown_ready():
    with st.spinner("过滤标的池并计算因子..."):
        def _scan():
            spot = dl.get_spot()
            uni = dl.filter_universe(spot)
            pool = dl.build_stock_pool(uni, max_pool=pool_size)
            return F.score_pool(pool, spot, market_score=_thermo_cached()["market_score"]).head(20)
        top = _guard(_scan)
        if top is not None:
            st.session_state["top"] = top
if "top" in st.session_state:
    _stale_warn()
    st.dataframe(st.session_state["top"].round(3), width="stretch", hide_index=True)
    st.caption("⚠️ Top名单仅供研究观测，题材情绪因子拥挤度高、成分股进出频繁。")
