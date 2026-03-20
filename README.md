# 🤖 美股量化交易入门项目

基于 Python + Alpaca API 的量化交易学习项目，从数据获取、策略回测到模拟盘交易的完整流程。

## 项目结构

```
quant-trading/
├── config.py                 # 🔑 配置文件（API Key、策略参数）
├── 01_account_check.py       # 📊 账户连接测试
├── 02_market_data.py         # 📡 行情数据获取（yfinance 免费数据）
├── 03_backtest_strategy.py   # 📈 策略回测（3种策略 + 参数优化）
├── 04_paper_trading.py       # 💹 模拟盘交易（交互式操盘台）
├── 05_auto_trader.py         # 🤖 自动化交易机器人
└── README.md                 # 📖 使用说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install alpaca-py yfinance pandas matplotlib backtesting
```

### 2. 配置 API Key

1. 注册 [Alpaca](https://app.alpaca.markets) 账号（免费）
2. 登录后切换到 **Paper Trading**（左上角下拉菜单）
3. 点击 **Generate New Key**，获取 API Key 和 Secret
4. 编辑 `config.py`，填入你的 Key：

```python
API_KEY = "PK..."        # 你的 API Key
API_SECRET = "abc..."    # 你的 Secret Key
```

### 3. 按顺序运行

```bash
# Step 1: 测试 API 连接
python 01_account_check.py

# Step 2: 获取行情数据（不需要 API Key）
python 02_market_data.py              # 默认 AAPL
python 02_market_data.py TSLA         # 指定股票
python 02_market_data.py NVDA 1y      # 指定时间范围

# Step 3: 策略回测（不需要 API Key）
python 03_backtest_strategy.py        # AAPL 默认参数
python 03_backtest_strategy.py TSLA   # 回测特斯拉
python 03_backtest_strategy.py NVDA 5 20  # 自定义均线参数

# Step 4: 模拟盘交易（需要 API Key）
python 04_paper_trading.py            # 交互式操盘台

# Step 5: 自动化交易（需要 API Key）
python 05_auto_trader.py AAPL         # 分析苹果并自动执行
```

## 包含的策略

| 策略 | 原理 | 文件 |
|------|------|------|
| **双均线交叉** | 短期均线上穿长期均线买入，下穿卖出 | `03_backtest_strategy.py` |
| **RSI 超买超卖** | RSI<30 买入，RSI>70 卖出 | `03_backtest_strategy.py` |
| **布林带** | 触及下轨买入，触及上轨卖出 | `03_backtest_strategy.py` |
| **综合信号** | 均线+RSI 多指标综合打分 | `05_auto_trader.py` |

## 技术栈

- **Alpaca API** — 零佣金券商 API，支持模拟盘和实盘
- **yfinance** — Yahoo Finance 免费行情数据
- **backtesting.py** — 轻量级策略回测框架
- **pandas / matplotlib** — 数据分析和可视化

## 注意事项

- ⚠️ 本项目仅供学习，不构成投资建议
- 📊 回测表现不代表未来收益
- 🧪 请先用 Paper Trading（模拟盘）充分测试
- 🔑 永远不要在代码中硬编码真实账户的 API Key
- 🕐 美股交易时间: 美东时间 9:30-16:00（北京时间 21:30-04:00）

## 下一步学习

1. 在 `03_backtest_strategy.py` 中添加自己的策略类
2. 尝试更多股票标的和参数组合
3. 学习 [QuantConnect](https://www.quantconnect.com) 做更专业的回测
4. 了解 [微软 Qlib](https://github.com/microsoft/qlib) 做 AI/ML 策略
