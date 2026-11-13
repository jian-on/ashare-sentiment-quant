# -*- coding: utf-8 -*-
"""
factors.py — 模块三A：A股短线因子库（全部为横截面打分，非涨跌预测）

改动点：
- 因子方向统一为"分越高越好"，与 config.FACTOR_WEIGHTS 配套
- 所有因子只用T日及之前数据，T+1执行，杜绝未来函数（规则3）

风险点 / 过拟合隐患：
- 反转/换手类因子在题材股情绪极端期会失效（涨得好的继续涨，超跌的继续跌），
  这是A股动量-反转混合结构，需在回测中分市场情绪状态检验
"""
import numpy as np
import pandas as pd

import config as C


def stock_factors(hist: pd.DataFrame) -> dict | None:
    """
    单只股票短线因子（输入前复权日K，按date升序）。
    返回 {因子名: 原始值}，方向未统一，统一在 score_pool 里做。
    """
    if hist is None or len(hist) < 25:
        return None
    close = hist["close"].astype(float).reset_index(drop=True)
    n = len(close)

    # 1) 5日反转：近5日涨幅取负（超跌加分）
    ret_5d = close.iloc[-1] / close.iloc[-6] - 1 if n >= 6 else 0.0
    reversal_5d = -ret_5d

    # 2) 20日价格位置：0=区间最低 1=区间最高（低位加分）
    w20 = close.tail(20)
    rng = w20.max() - w20.min()
    pos_in_20d = 1 - (close.iloc[-1] - w20.min()) / rng if rng > 0 else 0.5

    # 3) 低波动：20日对数收益率标准差取负
    logret = np.log(close / close.shift(1)).dropna()
    low_vol = -logret.tail(20).std() if len(logret) >= 10 else 0.0

    # 4) 量能趋势：近5日均量/近20日均量（温和放量>1，极端放量>2.5反而扣分在打分层处理）
    vol = hist["volume"].astype(float).reset_index(drop=True)
    vol_ratio_5_20 = vol.tail(5).mean() / max(vol.tail(20).mean(), 1)

    # 5) 换手率分位（近60日内当前换手率的分位，越高越活跃）
    if "turnover" in hist.columns:
        to = pd.to_numeric(hist["turnover"], errors="coerce").dropna()
        turnover_pct = (to < to.iloc[-1]).mean() if len(to) > 20 else 0.5
    else:
        turnover_pct = 0.5

    return {"reversal_5d": reversal_5d, "pos_in_20d": pos_in_20d, "low_vol": low_vol,
            "vol_ratio_5_20": vol_ratio_5_20, "turnover_pct": turnover_pct}


def winsorize_z(s: pd.Series, k: float = 3.0) -> pd.Series:
    """去极值(±3σ截断)后z标准化——横截面标准处理"""
    s = s.astype(float)
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s * 0
    s = s.clip(mu - k * sd, mu + k * sd)
    return (s - s.mean()) / s.std()


def score_pool(pool: dict[str, pd.DataFrame], spot: pd.DataFrame,
               sentiment_map: dict[str, float] | None = None,
               market_score: float | None = None) -> pd.DataFrame:
    """
    全股票池因子打分。
    pool: {code: 前复权df}; spot: 实时快照; sentiment_map: {code: 舆情分}
    market_score: 市场情绪温度(-1~1)，同一截面常数，不参与横截面z但进加权
    返回 DataFrame(code, 名称, 各因子z分, composite)
    """
    rows = {}
    for code, hist in pool.items():
        f = stock_factors(hist)
        if f is None:
            continue
        row = spot[spot["代码"].astype(str) == code].iloc[0]
        f["code"] = code
        f["名称"] = str(row["名称"])
        f["量比_raw"] = pd.to_numeric(row.get("量比"), errors="coerce")
        rows[code] = f
    df = pd.DataFrame(rows).T

    # 因子方向统一与标准化
    df["reversal_5d"] = winsorize_z(df["reversal_5d"])          # 越负(超跌)越加分？已是负向原始值，z后高=超跌多
    df["pos_in_20d"] = 1 - winsorize_z(df["pos_in_20d"])        # 位置低加分
    df["low_vol"] = winsorize_z(df["low_vol"])                  # 原始已是负波动，z后高=波动小
    vr = winsorize_z(df["vol_ratio_5_20"])
    df["vol_trend"] = np.where(df["vol_ratio_5_20"] > 2.5, -np.abs(vr), vr)  # 极端放量=情绪顶部信号，扣分
    df["turnover_z"] = winsorize_z(df["turnover_pct"])
    df["volume_ratio"] = winsorize_z(pd.to_numeric(df["量比_raw"], errors="coerce"))
    df["sentiment"] = pd.Series({c: sentiment_map.get(c, 0.0) for c in df.index}) if sentiment_map else 0.0
    df["market_beta"] = float(market_score) if market_score is not None else 0.0

    weights = dict(C.FACTOR_WEIGHTS)
    df["composite"] = sum(df[k].astype(float) * w for k, w in weights.items() if k in df.columns)
    cols = ["code", "名称", "reversal_5d", "pos_in_20d", "low_vol", "vol_trend",
            "turnover_z", "volume_ratio", "sentiment", "market_beta", "composite"]
    return df[cols].sort_values("composite", ascending=False).reset_index(drop=True)
