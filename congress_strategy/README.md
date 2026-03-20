# 🏛️ 国会议员跟单量化策略系统

**Congressional Trading Follow Strategy System**

基于美国国会议员公开交易数据的量化跟单策略。利用 STOCK Act (2012) 强制披露的国会议员交易记录，通过多维度信号评分筛选高置信度交易机会。

---

## 📊 策略原理

美国法律要求国会议员在 45 天内公开披露任何股票交易。研究显示，国会议员的交易表现显著跑赢市场：

- Congress buy strategy CAGR **37.4%** since 2020
- 民主党 **+31%**，共和党 **+26%** (2024) vs S&P 500 +24.9%

本系统自动获取这些数据，评分筛选，并通过 Alpaca API 自动执行。

## 🚀 快速开始

### 安装依赖

```bash
pip install requests pandas yfinance alpaca-py
```

### 快速扫描信号

```bash
# 扫描最近 90 天的高评分信号
python congress_strategy/main.py --scan

# 运行历史回测 (持仓30天)
python congress_strategy/main.py --backtest --hold 30

# 模拟交易执行
python congress_strategy/main.py --trade

# 查看投资组合
python congress_strategy/main.py --portfolio

# 交互式菜单
python congress_strategy/main.py
```

### Python API 调用

```python
from congress_strategy.data_fetcher import fetch_congress_trades
from congress_strategy.signal_scorer import generate_signals
from congress_strategy.backtester import run_backtest
from congress_strategy.trader import execute_strategy

# 1. 获取数据
df = fetch_congress_trades(days=90)

# 2. 生成信号
signals = generate_signals(df, min_score=50, top_n=10)

# 3. 回测验证
result = run_backtest(scored_df, hold_days=30, min_signal_score=50)

# 4. 模拟交易
execute_strategy(signals, dry_run=True)
```

## 📐 信号评分系统 (0-100分)

| 维度 | 分值 | 逻辑 |
|------|------|------|
| **交易金额** | 0-25 | 金额越大 → 议员信心越强 |
| **委员会相关性** | 0-20 | 议员所在委员会与股票行业相关 → 信息优势 |
| **集群信号** | 0-25 | 多位议员同时买入同一股票 → 最强信号 |
| **申报速度** | 0-15 | 快速申报 → 交易更有参考价值 |
| **历史成功率** | 0-15 | 该议员过往胜率高 → 更值得跟随 |

### 仓位策略
- 评分 ≥ 80: 满仓 (10% 仓位)
- 评分 60-79: 半仓 (5% 仓位)
- 评分 < 60: 不操作

## 🛡️ 风险控制

| 规则 | 阈值 | 说明 |
|------|------|------|
| 单只股票上限 | 10% | 防止集中风险 |
| 行业集中度上限 | 30% | 分散行业风险 |
| 个股止损 | -8% | 及时止损 |
| 回撤暂停 | -15% | 超过阈值暂停所有交易 |
| 最大同时持仓 | 10只 | 管理仓位数量 |

## 📁 文件结构

```
congress_strategy/
├── __init__.py          # 包初始化
├── main.py              # 主控制台 (命令行 + 交互式菜单)
├── data_fetcher.py      # 数据获取模块 (QuiverQuant + GitHub)
├── signal_scorer.py     # 信号评分引擎
├── backtester.py        # 历史回测系统
├── trader.py            # Alpaca 自动交易模块
├── data/                # 缓存数据 (自动生成)
├── logs/                # 交易日志 (自动生成)
└── README.md            # 本文件
```

## 📊 数据源

| 数据源 | 类型 | 费用 | 覆盖范围 |
|--------|------|------|----------|
| QuiverQuant 公开页面 | 实时 | 免费 | 最近300笔交易 |
| Senate Stock Watcher (GitHub) | 历史 | 免费 | 2012-2020 |
| Alpaca Markets | 交易执行 | 免费 | Paper Trading |

## ⚠️ 免责声明

- 本系统仅用于 **学习和研究** 目的
- Paper Trading (模拟盘) 不涉及真实资金
- 过往表现不代表未来收益
- 实际使用前请充分理解风险并做好尽职调查
- 国会议员的交易披露存在 45 天延迟，信息时效性有限
