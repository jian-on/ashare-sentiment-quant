# -*- coding: utf-8 -*-
"""
factors.py — A股短线因子库（横截面打分，非涨跌预测）

2026-11-13 多智能体审计后的修订，详见 RESEARCH_2026-11-13.md：
  1. 修复 pos_in_20d 双重反向：stock_factors 已把方向统一为「低位=高分」，
     score_pool 原来的 `1 - z(...)` 又翻了一次，实盘验证 Top10 因此系统性追高弃低。
  2. winsorize_z 改用分位数截断：原 ±3σ 的边界由含异常值的均值/标准差计算，
     会被异常值自身撑大，去极值形同虚设；同时修 NaN 传播与全 NaN 分支。
  3. sentiment 做横截面标准化：原实现把 [-1,1] 且集中在 ±0.1~0.3 的原始分
     直接与 z 分数加权，名义 0.25/0.30 的最高权重实际贡献被压到 1/5 以下。
  4. 启用 vol_trend：原来算了「极端放量扣分」却不在 FACTOR_WEIGHTS 里，是死代码。
  5. turnover_z 改由 spot 换手率横截面分位生成：腾讯K线无历史换手率，
     原实现退化为常数 0.5 → z 后全 0，15% 权重静默空转。
  6. 移除 market_beta 常数因子：同一截面所有股票同值，对排序零贡献，
     只会让 composite 绝对值随市场温度漂移、跨日不可比。保留列仅供诊断。
  7. 新增有效权重归一化：横截面标准差≈0 的因子自动剔除并把剩余权重重归一到 1.0，
     杜绝「名义 1.0、实际只有 0.5 在起作用」，诊断信息可回传 UI。

规则3 仍然成立：所有因子只用 T 日及之前数据，T+1 执行，无未来函数。
已知口径问题（未改，见研究报告）：盘中未收盘 K 线会让日内因子漂移；
收盘后运行不受影响。
"""
import numpy as np
import pandas as pd

import config as C


def stock_factors(hist: pd.DataFrame) -> dict | None:
    """
    单只股票短线因子（输入前复权日K，按date升序）。
    方向已在原始定义处统一为「分越高越好」，score_pool 只做标准化，不再翻转。
    turnover_pct 无历史换手率时返回 None，由 score_pool 用 spot 横截面分位补齐。
    """
    if hist is None or len(hist) < 25:
        return None
    close = hist["close"].astype(float).reset_index(drop=True)
    n = len(close)

    # 1) 5日反转：近5日涨幅取负 —— 值越高 = 跌得越多（超跌加分）
    ret_5d = close.iloc[-1] / close.iloc[-6] - 1 if n >= 6 else 0.0
    reversal_5d = -ret_5d

    # 2) 20日价格位置：原始 (close-min)/rng 是 0=最低 1=最高；
    #    取 1- 之后变成「越接近区间低位，值越高」（低位加分）
    w20 = close.tail(20)
    rng = w20.max() - w20.min()
    pos_in_20d = 1 - (close.iloc[-1] - w20.min()) / rng if rng > 0 else 0.5

    # 3) 低波动：20日对数收益率标准差取负 —— 值越高 = 波动越小
    logret = np.log(close / close.shift(1)).dropna()
    low_vol = -logret.tail(20).std() if len(logret) >= 10 else 0.0

    # 4) 量能趋势：近5日均量/近20日均量（方向调整在 score_pool 做）
    vol = hist["volume"].astype(float).reset_index(drop=True)
    vol_ratio_5_20 = vol.tail(5).mean() / max(vol.tail(20).mean(), 1)

    # 5) 换手率分位：腾讯K线不含 turnover → 显式 None，不用假常数冒充「中性」
    if "turnover" in hist.columns:
        to = pd.to_numeric(hist["turnover"].tail(60), errors="coerce").dropna()
        if len(to) > 20:
            turnover_pct = float((to < to.iloc[-1]).mean())
        else:
            turnover_pct = None
    else:
        turnover_pct = None

    return {"reversal_5d": reversal_5d, "pos_in_20d": pos_in_20d, "low_vol": low_vol,
            "vol_ratio_5_20": vol_ratio_5_20, "turnover_pct": turnover_pct}


def winsorize_z(s: pd.Series, k: float = 3.0) -> pd.Series:
    """
    稳健去极值 + z 标准化。

    原实现用「含异常值的原始 mu/sd」定 ±kσ 边界，异常值会把边界撑到自己之外，
    截断失效（实盘验证：格力量比 +3.11 完全没被截）。现改为两级保护：
      ① 1%/99% 分位数截断（不受异常值自身撑大影响）
      ② 标准化后再硬截断到 ±k，保证输出确实落在名义区间内
    缺失值填中位数；零方差/全 NaN 返回全 0（中性，不会静默丢票）。
    """
    s = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    med = s.median()
    s = s.fillna(0.0 if pd.isna(med) else float(med))
    if len(s) >= 20:
        s = s.clip(float(s.quantile(0.01)), float(s.quantile(0.99)))
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-abs(k), abs(k))


def _effective_weights(df: pd.DataFrame, weights: dict) -> tuple[dict, list[str]]:
    """
    剔除横截面无区分度（std≈0）的因子，剩余权重按合计重归一到 1.0。
    这是「实际参与排序的权重」可观测化的关键：修复前 0.50 权重静默空转。
    """
    live, dropped = {}, []
    for k, w in weights.items():
        if k not in df.columns:
            dropped.append(f"{k}(缺列)")
            continue
        col = pd.to_numeric(df[k], errors="coerce").astype(float)
        if float(col.std(ddof=0)) < 1e-8 or col.isna().all():
            dropped.append(k)
            continue
        live[k] = float(w)
    total = sum(live.values())
    if total > 0 and abs(total - 1.0) > 1e-9:
        live = {k: v / total for k, v in live.items()}
    return live, dropped


def score_pool(pool: dict[str, pd.DataFrame], spot: pd.DataFrame,
               sentiment_map: dict[str, float] | None = None,
               market_score: float | None = None,
               weights: dict | None = None,
               return_diag: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    全股票池因子打分（横截面相对排序，不是涨跌预测）。

    pool: {code: 前复权df}; spot: 实时快照
    sentiment_map: {code: 舆情原始分}；内部做横截面标准化，缺失记中性并计入覆盖率
    market_score: 市场情绪温度(-1~1)，仅用于诊断/状态判定，不再作为常数因子加权
    weights: 覆盖权重（供市场状态自适应实验使用）；None 用 config.FACTOR_WEIGHTS
    return_diag: True 时返回 (df, 诊断dict)，诊断含有效权重、被剔除因子、舆情覆盖率
    """
    spot_code = spot["代码"].astype(str)
    spot_idx = spot.set_index(spot_code)
    rows = {}
    for code, hist in pool.items():
        f = stock_factors(hist)
        if f is None:
            continue
        if code not in spot_idx.index:
            continue
        row = spot_idx.loc[code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        f["code"] = code
        f["名称"] = str(row.get("名称", code))
        f["量比_raw"] = pd.to_numeric(row.get("量比"), errors="coerce")
        # 腾讯K线无历史换手率 → 用同一份快照的当日换手率，横截面分位在下面统一算
        f["换手率_raw"] = pd.to_numeric(row.get("换手率"), errors="coerce")
        f["asof_raw"] = row.get("asof")
        rows[code] = f
    if not rows:
        empty = pd.DataFrame(columns=["code", "名称", "composite"])
        return (empty, {"error": "无可用标的"}) if return_diag else empty

    df = pd.DataFrame(rows).T

    # ---- 方向统一 + 标准化（全部为 z 分数，均值 0 / 标准差 1）----
    df["reversal_5d"] = winsorize_z(df["reversal_5d"])       # 高 = 超跌多
    df["pos_in_20d"] = winsorize_z(df["pos_in_20d"])         # 高 = 区间低位（不再翻转）
    df["low_vol"] = winsorize_z(df["low_vol"])               # 高 = 波动小

    vr_raw = pd.to_numeric(df["vol_ratio_5_20"], errors="coerce").astype(float)
    vr = winsorize_z(vr_raw)
    df["vol_trend"] = np.where(vr_raw > 2.5, -np.abs(vr), vr)  # 极端放量 = 情绪顶部，扣分

    # 换手活跃度：优先历史换手率分位；缺失时用 spot 换手率的横截面分位
    hist_to = pd.to_numeric(df["turnover_pct"], errors="coerce")
    if hist_to.notna().sum() >= max(20, int(len(df) * 0.5)):
        df["turnover_z"] = winsorize_z(hist_to.fillna(hist_to.median()))
        turnover_src = "历史换手率分位"
    else:
        spot_to = pd.to_numeric(df["换手率_raw"], errors="coerce")
        df["turnover_z"] = winsorize_z(spot_to.rank(pct=True))
        turnover_src = "快照换手率横截面分位"
    df["volume_ratio"] = winsorize_z(df["量比_raw"])

    # 舆情：原始分 → 横截面 z；未打分者记中性 0，并显式统计覆盖率
    if sentiment_map:
        sent_raw = pd.Series({c: float(sentiment_map.get(c, 0.0)) for c in df.index})
        # 覆盖率 = 「有打分记录」占比；情绪分恰好 0.0 是中性，不能算未覆盖
        coverage = round(sum(1 for c in df.index if c in sentiment_map) / len(df), 4)
    else:
        sent_raw = pd.Series(0.0, index=df.index)
        coverage = 0.0
    df["sentiment"] = winsorize_z(sent_raw)

    # 仅诊断：市场温度是截面常数，绝不参与 composite
    df["market_beta"] = float(market_score) if market_score is not None else 0.0

    nominal = dict(weights or C.FACTOR_WEIGHTS)
    live_w, dropped = _effective_weights(df, nominal)
    df["composite"] = sum(
        pd.to_numeric(df[k], errors="coerce").astype(float).fillna(0.0) * w
        for k, w in live_w.items()
    )

    cols = ["code", "名称", "reversal_5d", "pos_in_20d", "low_vol", "vol_trend",
            "turnover_z", "volume_ratio", "sentiment", "market_beta", "composite"]
    out = df[cols].sort_values("composite", ascending=False).reset_index(drop=True)

    # 截面时点一致性校验：sources.py 零成本带出腾讯字段 30（每只股票最后成交时间）。
    # 只校验「实际进入打分的股票」——全A里停牌/僵死股的最后成交时间停留在早盘，
    # 把它们一起算跨度会误报 7 小时，但它们已被 filter_universe 剔除。
    asof_info = {"asof": None, "asof_span_sec": None, "asof_ok": None,
                 "asof_outliers": 0}
    if "asof_raw" in df.columns:
        av = pd.to_datetime(df["asof_raw"].astype(str), format="%Y%m%d%H%M%S",
                            errors="coerce").dropna()
        if not av.empty:
            # 收盘后各股「最后成交时间」相差几十秒是正常现象，不能按秒级相等判异常。
            # 真正的问题是跨交易日混合或时点跨度过大（>15分钟）。
            day = av.dt.normalize()
            main_day = day.mode().iloc[0] if not day.mode().empty else day.iloc[0]
            outliers = int((day != main_day).sum())
            span = float((av.max() - av.min()).total_seconds())
            ok = outliers == 0 and span <= 900.0
            asof_info = {"asof": av.max().strftime("%Y-%m-%d %H:%M:%S"),
                         "asof_span_sec": span, "asof_ok": bool(ok),
                         "asof_outliers": outliers}
            if C.STRICT_ASOF and not ok:
                print(f"[警告] 打分池行情时点跨 {outliers} 个交易日 / 跨度 {span:.0f}s，"
                      f"本次横截面可能不一致")

    diag = {
        "n_stocks": len(out),
        "weights_nominal": nominal,
        "weights_effective": live_w,
        "effective_weight_total": round(sum(live_w.values()), 4),
        "dropped_factors": dropped,
        "sentiment_coverage": round(coverage, 4),
        "turnover_source": turnover_src,
        "market_score": market_score,
        **asof_info,
    }
    return (out, diag) if return_diag else out
