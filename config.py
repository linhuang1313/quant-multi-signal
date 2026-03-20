"""
Alpaca API 配置文件
===================
在 https://app.alpaca.markets 的 Paper Trading 页面生成你的 API Key

操作步骤：
1. 登录 Alpaca Dashboard
2. 左上角切换到 "Paper Trading"
3. 点击 "Generate New Key"
4. 将生成的 Key 和 Secret 填入下方
"""

# ============================================================
# 🔑 在这里填入你的 Alpaca Paper Trading API Keys
# ============================================================
API_KEY = "PKCTMFDROQEWG5ESKZB75ZEPZA"
API_SECRET = "DynBRwaD34metqn1FsnTiPvCBxdBMKkRwoothbxwxWw8"

# Paper Trading（模拟盘）的 Base URL
# 实盘交易时改为: https://api.alpaca.markets
BASE_URL = "https://paper-api.alpaca.markets"

# ============================================================
# 策略参数（可自行调整）
# ============================================================
# 双均线策略参数
SHORT_WINDOW = 10    # 短期均线天数
LONG_WINDOW = 30     # 长期均线天数

# 回测参数
BACKTEST_START = "2024-01-01"
BACKTEST_END = "2026-03-20"
INITIAL_CASH = 100000  # 初始资金 $100,000

# 默认标的
DEFAULT_SYMBOL = "AAPL"  # 苹果股票
