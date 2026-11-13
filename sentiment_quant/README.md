# A股题材情绪多因子选股终端（原型）

> ⚠️ 仅用于学习验证，禁止实盘重仓。题材/舆情类因子天然容易失效。

## 一键启动（推荐）

**双击桌面上的 `A股量化终端.bat`** —— 自动启动服务并打开浏览器，不依赖任何对话会话。

项目内也有同样的 `启动终端.bat`（相对路径，整个文件夹可搬移）和 `停止终端.bat`。

> 若浏览器打不开，请用 `http://127.0.0.1:8501` 而不是 `localhost`——
> 本机装有系统代理（127.0.0.1:50510），`localhost` 请求会被代理拦截返回 502。

## 模块结构
| 文件 | 职责 | 状态 |
|---|---|---|
| config.py | 全局参数（过滤阈值/因子权重/费用/缓存TTL） | ✅ |
| **sources.py** | **多源行情适配层：腾讯批量快照 + 并发前复权K线（解决东财限流的核心）** | ✅ 已实测 |
| data_loader.py | 标的四重过滤 + 前复权K线（腾讯主源 → 东财备源） | ✅ 已烟测 |
| sentiment.py | 舆情打分：通义千问API + 规则词典降级 | ✅ 降级模式已烟测 |
| factors.py | 短线因子库 + 横截面打分 | ✅ 已烟测 |
| market_sentiment.py | 市场情绪温度计（**基于快照本地计算，零额外请求**） | ✅ 已烟测 |
| watchlist.py | 自选股全面分析（四路数据融合） | ✅ 已烟测 |
| recommend.py | 市场分析驱动的个股参考清单 | ✅ 已烟测 |
| make_launcher.py | 生成桌面/项目内双击启动器（GBK 编码写 .bat） | ✅ |
| backtest.py | 滚动回测（手续费/滑点/训练验证样本外） | ⏳ 待开发 |
| app.py | Streamlit 单页面观测终端 | ✅ |

## 数据源架构：为什么东财会一直限流

| | 东财（旧方案） | 腾讯（现方案） |
|---|---|---|
| 全A快照 | 59 页分页请求 | **12 批批量请求，约 1 秒** |
| 单只K线 | 0.5s 强制节流 → 150只需 75 秒 | **4 线程并发，150 只约 5 秒** |
| 一次扫描总请求 | 200+ 次 | **约 15 次**（5分钟内重扫全命中缓存） |
| 结果 | 必被 IP 限流，5-40 分钟 | 正常使用不触发 |

**根本原因**：东财是"逐只拉取"，150 只候选池必须发 150 次请求，这跟调参数没关系，
是架构决定的。腾讯支持批量查询，请求数降一个数量级，才是真正的解法。
东财保留为备源（failover），不再是主链路。

### 三级保护机制（防止把自己玩死）
1. **缓存优先**：快照 60 秒 / K线 5 分钟 TTL，重复点击几乎零请求
2. **源健康状态**：某源被拦截后自动进入冷却（腾讯 3 分钟 / 东财 3 分钟），
   冷却期内**直接跳过不发请求**——这是关键，之前反复点击会把封禁从 5 分钟拖到 40 分钟
3. **WAF 熔断**：腾讯连续 15 只失败判定为整体拦截，立即中止整批而非硬扛

### 接口实测记录（2026-11-05 实盘）
```
qt.gtimg.cn 批量行情      HTTP 200，499 只 0.11s，字段含现价/涨跌幅/换手率/量比/总市值/涨停价
web.ifzq.gtimg.cn 前复权K线  HTTP 200，121 根 0.09s，含当日盘中K线
全市场 5574 只          26 批 → 0.7s（建立 active_codes.txt 后降至 12 批）
K线 120 只              4 线程 → 5s
```
**腾讯也有 WAF**（返回 501 跳转 waf.tencent.com），阈值远高于东财，
但短时间内连续跑多次全量扫描仍会触发。触发后需冷却 10-30 分钟。

## 环境（本机实测路径）
```
PY="C:/Users/27553/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
$PY sources.py                # 数据源烟测（快照+K线）
$PY sources.py --debug        # 打印原始字段，接口变更时排查
$PY data_loader.py            # 数据层烟测
$PY market_sentiment.py       # 温度计（本地计算）+ 财联社快讯
$PY watchlist.py --codes 002384,600519
$PY recommend.py              # 参考清单
$PY make_launcher.py          # 重新生成启动器
```

## 通义千问接入
```bash
export DASHSCOPE_API_KEY=sk-xxxx    # Windows CMD: set DASHSCOPE_API_KEY=sk-xxxx
```
配置后 sentiment.py 自动切换 qwen-plus（temperature=0.1，15条/批）。
未配置时自动降级规则词典，source 标签显式区分 `RULE_BASED` / `QWEN`，绝不冒充。

## 已知问题与限制（不隐瞒）
1. **腾讯K线不返回历史换手率/成交额** → `turnover_pct` 因子降级为中性 0.5。
   影响：换手活跃度因子暂失效（不是错误信号，是缺失）。补法：从快照换手率+分批历史重建。
2. **腾讯WAF**：短时高频会拦截，需冷却 10-30 分钟（已内置熔断，不会越拖越久）。
3. **股吧源不可用**：akshare 1.20.22 已移除 `stock_guba_em`，gbapi 返回空（需登录态），
   舆情现以个股新闻为主。调试方案见 sentiment.py 末尾注释。
4. `stock_news_em` 与 pandas 3.0 不兼容（pyarrow 拒绝 `\u3000` 正则）→ 已改 requests 直连上游。

## 接口失效排查三板斧
```python
import akshare as ak
print([f for f in dir(ak) if '关键词' in f.lower()])   # 1. 确认函数是否存在
df = ak.xxx(); print(list(df.columns)); print(df.head()) # 2. 确认字段名
# 3. 字段对不上 → 打开 akshare 源码 stock/ 目录下对应 py，找到上游 URL 用 requests 直连
```
自建源排查：`python sources.py --debug` 打印腾讯原始返回（含字段索引）。
