# -*- coding: utf-8 -*-
"""
regime.py — 市场情绪状态判定 + 自适应权重（实验性，默认关闭）

设计来源：2026-11-13 多智能体研究中的「市场状态研究员」角色。
依据 A 股动量-反转混合结构：高潮期动量占优（涨得好的继续涨），
退潮期反转/位置/量能同时失效（高位反杀 + 超跌继续跌），需提高防守权重。

⚠️ 重要声明：本模块全部阈值与权重是 2026-11-13 单日复盘归纳的**待验证假设**，
   未经本项目样本外回测证明有效，可能被证伪。因此 config.ENABLE_REGIME_ADAPT
   默认 False —— 关闭时模型一律使用 config.FACTOR_WEIGHTS 基准权重。
   开启前必须完成 backtest.py 的 point-in-time 存档 + purged walk-forward 检验。

判定只使用 market_sentiment.market_thermometer() 已提供的字段，不引入任何新数据源，
不含 T 日之后信息，T+1 执行；因此不会引入前视偏差（但会引入待验证的自由度）。
"""
import json
import os
import datetime as dt

import config as C

STATE_ORDER = ["冰点", "回暖", "高潮", "退潮"]
_STATE_FILE = os.path.join(C.CACHE_DIR, "regime_state.json")


def _num(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def classify_state(detail: dict, prev_state: str | None = None) -> str:
    """
    四态判定：冰点 / 回暖 / 高潮 / 退潮。
    detail = market_thermometer()["detail"]，字段缺失时保守回落到「回暖」。
    阈值刻意粗粒度、取整，避免细网格调参导致过拟合。
    """
    if not detail:
        return C.REGIME_DEFAULT_STATE
    s = _num(detail.get("market_score"))
    zt = _num(detail.get("涨停家数"))
    n_dt = _num(detail.get("跌停家数"))
    ban = detail.get("炸板率")
    ban = 0.5 if ban is None else _num(ban, 0.5)
    up = _num(detail.get("上涨"))
    down = _num(detail.get("下跌"))
    breadth = (up - down) / (up + down) if (up + down) > 0 else 0.0

    hot = s >= 0.30 and zt >= 70 and ban <= 0.20 and n_dt <= 5
    cold = s <= -0.30 and (zt <= 20 or s <= -0.50)
    bad = breadth <= -0.40 and (ban >= 0.35 or n_dt >= 30) and s <= 0

    if hot:
        return "高潮"
    if cold:
        return "冰点"
    if bad:
        return "退潮"
    if -0.30 < s < 0.30:
        # 中性区靠上一交易日方向记忆区分回暖/退潮，避免同分不同向被合并
        return "退潮" if prev_state in ("高潮", "退潮") else "回暖"
    return "回暖"


def load_prev_state() -> str | None:
    """读取上一交易日状态（本地 JSON，非外部数据源）"""
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("state")
    except (OSError, ValueError, AttributeError):
        return None


def save_state(state: str, market_score: float | None = None) -> None:
    """持久化当日状态，供次日滞回判定使用"""
    try:
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": dt.date.today().isoformat(), "state": state,
                       "market_score": market_score}, f, ensure_ascii=False)
    except OSError:
        pass


def weights_for(state: str) -> dict:
    """返回该状态的权重表；未知状态回退基准。所有权重表和为 1.0"""
    w = C.REGIME_WEIGHTS.get(state, C.FACTOR_WEIGHTS)
    total = sum(w.values())
    return {k: v / total for k, v in w.items()} if total > 0 else dict(C.FACTOR_WEIGHTS)


def resolve_weights(detail: dict, enabled: bool | None = None):
    """
    统一入口。返回 (weights, state, note)
    - 关闭时仍计算并回传 state（供 UI 展示观测值），但权重恒为基准
    - 开启时在日志里明确标注「实验性、未验证」
    """
    if enabled is None:
        enabled = C.ENABLE_REGIME_ADAPT
    prev = load_prev_state()
    state = classify_state(detail, prev_state=prev)
    if enabled:
        note = (f"⚠️ 实验模式：按「{state}」启用自适应权重，该规则未经样本外验证，"
                f"不得据此实盘或对外宣称有效")
        return weights_for(state), state, note
    return dict(C.FACTOR_WEIGHTS), state, f"观测状态「{state}」；自适应权重实验关闭，使用基准权重"
