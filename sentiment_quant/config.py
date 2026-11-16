# -*- coding: utf-8 -*-
"""
config.py — 全局参数配置

2026-11-13 多智能体审计后的修订（详见 RESEARCH_2026-11-13.md）：
- 移除 market_beta 权重：它是同一截面常数，对横截面排序零贡献，只让 composite 跨日漂移
- 启用 vol_trend：原代码算了「极端放量扣分」却从未进权重字典，是死代码
- turnover_z 改由 spot 换手率横截面分位生成（腾讯K线无历史换手率，原实现恒为 0）
- sentiment 改为横截面标准化后打分，且前置到候选集（原实现只对已入选 Top10 补分，是选择偏差）
- 新增「市场状态自适应权重」实验开关，默认关闭 —— 未通过样本外检验前不得当作有效改进

风险提示：
- min_amount / top_n 等参数调得越"漂亮"越容易过拟合，建议用滚动窗口验证后再调
- 所有权重为待验证假设，不是已被证明有效的最优参数
"""

# ========== 数据与缓存 ==========
CACHE_DIR = "data_cache"
HIST_DAYS = 120

# ========== 实时性口径 ==========
ALLOW_STALE_FALLBACK = False      # False=严格实时：任何源都失败时直接报错，绝不返回过期缓存
SPOT_TTL_SEC = 60
HIST_TTL_SEC = 300
MIN_REQUEST_INTERVAL = 0.5
STRICT_ASOF = True                # 快照带 asof 时间列；打分时校验截面一致性（见 sources.py f[30]）

# ========== 界面点击冷却 ==========
UI_COOLDOWN_SEC = 30
RATE_LIMIT_COOLDOWN = 120

# ========== 标的过滤 ==========
MIN_AMOUNT = 50_000_000
MIN_LISTED_DAYS = 60
EXCLUDE_ST = True
EXCLUDE_BJ = True
EXCLUDE_688 = False
INCLUDE_CHINEXT = True            # 2026-11-13 修复：原 filter_universe 只认 60/00/688，
                                  # 创业板 300/301 被静默全部剔除，与题材短线主线冲突

# 候选池选样方式：
#   "stratified" = 按当日成交额分位分 5 层等额抽样（固定种子，可复现），避免只选到大盘蓝筹
#   "top_amount" = 旧行为，成交额降序取前 N（存在「用当日结果选样」的自我指涉偏差）
POOL_SELECT_MODE = "stratified"
POOL_STRATA = 5
POOL_RANDOM_SEED = 42

# ========== 因子权重（修复后的基准，方向统一为"分越高越好"，和为 1.0）==========
FACTOR_WEIGHTS = {
    "reversal_5d":  0.18,   # 5日超跌反转
    "pos_in_20d":   0.12,   # 20日区间低位（2026-11-13 修复双重反向）
    "low_vol":      0.10,   # 低波动
    "vol_trend":    0.10,   # 量能趋势；>2.5 倍极端放量转为扣分（原来未启用）
    "turnover_z":   0.10,   # 换手活跃度；改由 spot 换手率横截面分位生成（原来恒 0）
    "volume_ratio": 0.10,   # 量比
    "sentiment":    0.30,   # 舆情情绪分（核心主线）；横截面 z 标准化后进入
}

# ========== 市场状态自适应权重（实验性，默认关闭）==========
# 依据：A股动量-反转混合结构 —— 高潮期动量占优、退潮期反转失效需防守。
# 这是 2026-11-13 单日复盘归纳的待验证假设，未经本项目样本外回测证明，可能被证伪。
ENABLE_REGIME_ADAPT = False
REGIME_DEFAULT_STATE = "回暖"
REGIME_WEIGHTS = {
    #              反转   位置   低波   量能趋势  换手   量比   舆情   = 合计
    "冰点": {"reversal_5d": 0.22, "pos_in_20d": 0.15, "low_vol": 0.18,
             "vol_trend": 0.06, "turnover_z": 0.06, "volume_ratio": 0.05, "sentiment": 0.28},
    "回暖": dict(FACTOR_WEIGHTS),   # 基准权重
    "高潮": {"reversal_5d": 0.06, "pos_in_20d": 0.04, "low_vol": 0.06,
             "vol_trend": 0.12, "turnover_z": 0.20, "volume_ratio": 0.18, "sentiment": 0.34},
    "退潮": {"reversal_5d": 0.05, "pos_in_20d": 0.05, "low_vol": 0.28,
             "vol_trend": 0.06, "turnover_z": 0.08, "volume_ratio": 0.08, "sentiment": 0.40},
}

# ========== 舆情打分范围 ==========
# 原实现先按无舆情分取 Top10、再只对这 10 只补舆情，舆情从不影响"谁入选"。
# 现在先取 SENTIMENT_COVERAGE_N 只确定性候选集，在集内全量补舆情后重排。
SENTIMENT_COVERAGE_N = 60         # ≥ 5×top_n；为 0 表示对全池打分（请求量大，慎用）
SENTIMENT_CACHE_TTL_SEC = 3600    # 同日重复扫描复用结果，避免反复请求上游

# ---- 舆情时点口径（2026-11-13 追加；是 point-in-time 可重放的前提）----
# 新闻按「发布日期」而非「抓取顺序」参与打分：过滤未来新闻与过期旧闻，
# 其余按半衰期加权。无发布日期的新闻一律丢弃（无法验证时点 = 不能用）。
NEWS_MAX_AGE_DAYS = 10            # 超过 10 天的新闻不进入短线情绪打分
NEWS_HALF_LIFE_DAYS = 3.0         # 3 天前的新闻权重减半
NEWS_MIN_WEIGHT = 0.05            # 权重低于此值直接丢弃，避免长尾旧闻拖累
NEWS_FUTURE_TOL_MIN = 5           # 上游时钟偏差容差（分钟），超过即视为未来新闻
NEWS_ARCHIVE = True               # 每只股票新闻明细按日期落盘，供历史重放
SENTIMENT_CACHE_VERSION = 2       # 缓存结构版本号；口径变更时必须失效重建

# ========== 回测参数（规则3：手续费/滑点/滚动窗口）==========
# 注意：backtest.py 已提供 point-in-time 存档与 purged walk-forward 骨架，
#       但当前尚无历史截面面板，任何权重/阈值都不得声称为"回测优化结果"。
COMMISSION = 0.00025
COMMISSION_MIN = 5.0
STAMP_TAX = 0.0005
SLIPPAGE = 0.001
TOP_N = 20
TRAIN_DAYS = 120
VALID_DAYS = 60
OOS_DAYS = 60
EMBARGO_DAYS = 5                  # 标签为 T+1~T+5 超额，窗口间需 5 日隔离防泄漏
MIN_STATE_SAMPLES = 20            # 单状态每窗口至少 20 个截面，不足则回退基准权重
# 板块分层检验（2026-11-16 追加）：10cm 主板与 20cm 创业板/科创板涨跌幅限制不同，
# low_vol、量能类因子的横截面分布会混合；必须分板块看 IC 是否同向，否则可能
# 出现「全样本 IC 为正但某一板块为负」被掩盖。只做诊断，不据此自动调权。
BOARD_20CM_PREFIXES = ("300", "301", "688")
MIN_BOARD_SAMPLES = 20            # 单板块 OOS 截面不足 20 个不出结论
# 固定外部基准：中证全指覆盖全A，不受本项目「候选池选样」影响。
# 用它的 T+1 开盘 → T+5 收盘收益做真实组合超额，池内相对收益只用于衡量排序能力。
BENCHMARK_SYMBOL = "sh000985"
BENCHMARK_LABEL = "中证全指(000985)"

# 收盘闸门（2026-11-16 追加）：本机为北京时间，早于此时刻一律拒绝写 PIT 存档。
# 理由：腾讯日K末根在盘中是动态值，且 PIT 存档不可变，一旦写入将永久污染当日截面。
# 定时任务固定在 15:50，正常路径不受影响；此闸门只拦截补跑/手动执行落在盘中的情形。
# 确需在盘中复盘时用 `--archive --force` 显式绕过，风险自负。
ARCHIVE_MIN_HHMM = "15:00"

# PIT 存档口径版本（2026-11-16 追加）。
# 口径一旦变更必须 +1，否则「旧因子值 + 新收益标签」会混进同一面板却无法区分：
#   v1 = 11-13 初版：舆情 sort=default 无时点过滤、无 score_source、无外部基准
#   v2 = A1/A2 后  ：舆情时点过滤+半衰期衰减+明细落库、分组标准化、中证全指双基准
#   v3 = A4/A5 后  ：新增 board（10cm/20cm 分层）与 quality_json（每日数据质量留痕）
#                    选样口径不变（当日成交额分 5 层等额抽样，固定 seed）
# 旧版本文件保留在磁盘供审计，但不进入回测面板、也不再补标签。
PIT_SCHEMA_VERSION = 3
