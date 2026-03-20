"""
03 - 双均线交叉回测策略
========================
经典的量化入门策略：
- 短期均线上穿长期均线 → 买入信号（金叉）
- 短期均线下穿长期均线 → 卖出信号（死叉）

使用 backtesting.py 框架进行历史回测

运行: python 03_backtest_strategy.py
      python 03_backtest_strategy.py TSLA          # 指定股票
      python 03_backtest_strategy.py NVDA 5 20     # 指定均线参数
"""

import sys
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from config import SHORT_WINDOW, LONG_WINDOW, INITIAL_CASH, BACKTEST_START, BACKTEST_END


class SmaCross(Strategy):
    """
    双均线交叉策略

    - n1: 短期均线周期（默认10天）
    - n2: 长期均线周期（默认30天）

    当短期均线上穿长期均线时全仓买入
    当短期均线下穿长期均线时全仓卖出
    """
    n1 = SHORT_WINDOW  # 短期均线
    n2 = LONG_WINDOW   # 长期均线

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), close, name=f'SMA{self.n1}')
        self.sma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), close, name=f'SMA{self.n2}')

    def next(self):
        # 金叉：短均线上穿长均线 → 买入
        if crossover(self.sma1, self.sma2):
            self.buy()
        # 死叉：短均线下穿长均线 → 卖出
        elif crossover(self.sma2, self.sma1):
            self.sell()


class RsiStrategy(Strategy):
    """
    RSI 超买超卖策略

    - rsi_period: RSI 计算周期（默认14天）
    - oversold: 超卖阈值（默认30，低于此值买入）
    - overbought: 超买阈值（默认70，高于此值卖出）
    """
    rsi_period = 14
    oversold = 30
    overbought = 70

    def init(self):
        close = pd.Series(self.data.Close)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        self.rsi = self.I(lambda: 100 - (100 / (1 + rs)), name='RSI')

    def next(self):
        if self.rsi[-1] < self.oversold:
            self.buy()
        elif self.rsi[-1] > self.overbought:
            self.sell()


class BollingerStrategy(Strategy):
    """
    布林带策略

    - bb_period: 布林带周期（默认20天）
    - bb_std: 标准差倍数（默认2）

    价格触及下轨 → 买入（超卖反弹）
    价格触及上轨 → 卖出（超买回落）
    """
    bb_period = 20
    bb_std = 2

    def init(self):
        close = pd.Series(self.data.Close)
        self.mid = self.I(lambda: close.rolling(self.bb_period).mean(), name='BB_Mid')
        std = close.rolling(self.bb_period).std()
        self.upper = self.I(lambda: close.rolling(self.bb_period).mean() + self.bb_std * std, name='BB_Upper')
        self.lower = self.I(lambda: close.rolling(self.bb_period).mean() - self.bb_std * std, name='BB_Lower')

    def next(self):
        if self.data.Close[-1] < self.lower[-1]:
            self.buy()
        elif self.data.Close[-1] > self.upper[-1]:
            self.sell()


def get_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """获取回测用的历史数据"""
    print(f"📡 获取 {symbol} 数据 ({start} → {end})...")
    df = yf.download(symbol, start=start, end=end)
    if hasattr(df.columns, 'droplevel'):
        df.columns = df.columns.droplevel(1)
    df.index.name = None
    print(f"✅ 获取到 {len(df)} 个交易日数据")
    return df


def run_backtest(df: pd.DataFrame, strategy_class, symbol: str, **kwargs):
    """运行回测并输出结果"""
    bt = Backtest(
        df,
        strategy_class,
        cash=INITIAL_CASH,
        commission=0.001,      # 0.1% 手续费（虽然Alpaca免佣，但留点滑点）
        exclusive_orders=True,  # 同时只能有一个方向的持仓
        trade_on_close=True,    # 以收盘价成交
    )

    # 运行回测
    stats = bt.run(**kwargs)

    strategy_name = strategy_class.__name__
    print(f"\n{'=' * 60}")
    print(f"📊 回测结果: {strategy_name} on {symbol}")
    print(f"{'=' * 60}")
    print(f"  策略收益率:    {stats['Return [%]']:.2f}%")
    print(f"  买入持有收益:  {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"  最大回撤:      {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  夏普比率:      {stats['Sharpe Ratio']:.4f}" if stats['Sharpe Ratio'] else "  夏普比率:      N/A")
    print(f"  交易次数:      {stats['# Trades']}")
    print(f"  胜率:          {stats['Win Rate [%]']:.2f}%" if stats['Win Rate [%]'] else "  胜率:          N/A")
    print(f"  最终资产:      ${stats['Equity Final [$]']:,.2f}")
    print(f"{'=' * 60}")

    # 保存回测图表
    chart_path = f"backtest_{symbol}_{strategy_name}.html"
    bt.plot(filename=chart_path, open_browser=False)
    print(f"📈 交互式图表已保存: {chart_path}")

    return stats, bt


def compare_strategies(df: pd.DataFrame, symbol: str):
    """对比多种策略的表现"""
    strategies = {
        "双均线交叉 (SMA Cross)": (SmaCross, {}),
        "RSI 超买超卖": (RsiStrategy, {}),
        "布林带策略": (BollingerStrategy, {}),
    }

    results = []
    for name, (strat_class, params) in strategies.items():
        bt = Backtest(df, strat_class, cash=INITIAL_CASH,
                     commission=0.001, exclusive_orders=True, trade_on_close=True)
        stats = bt.run(**params)
        results.append({
            "策略": name,
            "收益率(%)": round(stats['Return [%]'], 2),
            "买入持有(%)": round(stats['Buy & Hold Return [%]'], 2),
            "最大回撤(%)": round(stats['Max. Drawdown [%]'], 2),
            "夏普比率": round(stats['Sharpe Ratio'], 4) if stats['Sharpe Ratio'] else 'N/A',
            "交易次数": stats['# Trades'],
            "胜率(%)": round(stats['Win Rate [%]'], 2) if stats['Win Rate [%]'] else 'N/A',
        })

    comparison = pd.DataFrame(results)
    print(f"\n{'=' * 80}")
    print(f"📊 策略对比: {symbol}")
    print(f"{'=' * 80}")
    print(comparison.to_string(index=False))
    print(f"{'=' * 80}")

    return comparison


def optimize_sma(df: pd.DataFrame, symbol: str):
    """优化双均线策略的参数"""
    print(f"\n🔍 正在优化 {symbol} 的双均线参数...")

    bt = Backtest(df, SmaCross, cash=INITIAL_CASH,
                 commission=0.001, exclusive_orders=True, trade_on_close=True)

    # 网格搜索最优参数
    stats = bt.optimize(
        n1=range(5, 25, 5),     # 短期均线：5, 10, 15, 20
        n2=range(20, 60, 10),   # 长期均线：20, 30, 40, 50
        maximize='Sharpe Ratio',
        constraint=lambda p: p.n1 < p.n2  # 短期 < 长期
    )

    print(f"\n✅ 最优参数:")
    print(f"  短期均线: {stats._strategy.n1} 天")
    print(f"  长期均线: {stats._strategy.n2} 天")
    print(f"  优化后收益率: {stats['Return [%]']:.2f}%")
    print(f"  优化后夏普比率: {stats['Sharpe Ratio']:.4f}" if stats['Sharpe Ratio'] else "")

    return stats


if __name__ == "__main__":
    # 命令行参数
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    n1 = int(sys.argv[2]) if len(sys.argv) > 2 else SHORT_WINDOW
    n2 = int(sys.argv[3]) if len(sys.argv) > 3 else LONG_WINDOW

    # 获取数据
    df = get_data(symbol, BACKTEST_START, BACKTEST_END)

    if df.empty:
        print("❌ 没有数据可回测")
        sys.exit(1)

    # 1. 运行双均线策略
    print("\n" + "🔸" * 30)
    print("  策略 1: 双均线交叉")
    print("🔸" * 30)
    run_backtest(df, SmaCross, symbol, n1=n1, n2=n2)

    # 2. 策略对比
    print("\n" + "🔸" * 30)
    print("  三种策略对比")
    print("🔸" * 30)
    compare_strategies(df, symbol)

    # 3. 参数优化
    print("\n" + "🔸" * 30)
    print("  参数优化（寻找最佳均线组合）")
    print("🔸" * 30)
    optimize_sma(df, symbol)
