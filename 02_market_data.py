"""
02 - 行情数据获取
==================
使用 yfinance 获取免费历史数据，支持任意美股标的
也演示了如何用 Alpaca API 获取实时数据

运行: python 02_market_data.py
      python 02_market_data.py TSLA     # 指定股票
      python 02_market_data.py NVDA 1y  # 指定股票和时间范围
"""

import sys
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_stock_data(symbol: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    """
    使用 yfinance 获取股票历史数据（免费，无需 API Key）

    参数:
        symbol: 股票代码，如 AAPL, TSLA, NVDA, GOOGL, MSFT
        period: 时间范围 - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max

    返回:
        DataFrame: 包含 Open, High, Low, Close, Volume 等列
    """
    print(f"📡 正在获取 {symbol} 的历史数据（{period}）...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        print(f"❌ 无法获取 {symbol} 的数据，请检查股票代码是否正确")
        return pd.DataFrame()

    print(f"✅ 获取到 {len(df)} 条数据")
    print(f"   时间范围: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   最新收盘价: ${df['Close'].iloc[-1]:.2f}")
    print(f"   期间最高价: ${df['High'].max():.2f}")
    print(f"   期间最低价: ${df['Low'].min():.2f}")

    return df


def plot_stock_chart(df: pd.DataFrame, symbol: str, save_path: str = None):
    """绘制股票K线图和成交量"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    fig.suptitle(f'{symbol} Stock Price Chart', fontsize=16, fontweight='bold')

    # 价格图
    ax1.plot(df.index, df['Close'], color='#2196F3', linewidth=1.5, label='Close Price')
    ax1.fill_between(df.index, df['Low'], df['High'], alpha=0.1, color='#2196F3')

    # 添加均线
    if len(df) >= 20:
        ma20 = df['Close'].rolling(window=20).mean()
        ax1.plot(df.index, ma20, color='#FF9800', linewidth=1, label='MA20', linestyle='--')
    if len(df) >= 50:
        ma50 = df['Close'].rolling(window=50).mean()
        ax1.plot(df.index, ma50, color='#F44336', linewidth=1, label='MA50', linestyle='--')

    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 成交量图
    colors = ['#4CAF50' if df['Close'].iloc[i] >= df['Open'].iloc[i]
              else '#F44336' for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7, width=1)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    path = save_path or f'{symbol}_chart.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"📊 图表已保存: {path}")
    plt.close()
    return path


def get_multiple_stocks(symbols: list, period: str = "1y") -> pd.DataFrame:
    """
    获取多只股票数据并对比收益率

    参数:
        symbols: 股票代码列表，如 ["AAPL", "GOOGL", "MSFT"]
        period: 时间范围

    返回:
        DataFrame: 各股票的归一化收益率
    """
    print(f"\n📡 正在获取 {len(symbols)} 只股票的数据...")
    all_data = {}

    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if not df.empty:
            # 归一化到 100（方便对比）
            all_data[symbol] = (df['Close'] / df['Close'].iloc[0]) * 100

    result = pd.DataFrame(all_data)
    print(f"✅ 数据获取完成")
    return result


def plot_comparison(df: pd.DataFrame, save_path: str = "comparison_chart.png"):
    """绘制多只股票收益对比图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('Stock Performance Comparison (Normalized to 100)',
                 fontsize=16, fontweight='bold')

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
              '#00BCD4', '#795548', '#607D8B']

    for i, col in enumerate(df.columns):
        color = colors[i % len(colors)]
        ax.plot(df.index, df[col], label=col, linewidth=2, color=color)
        # 标注最终收益
        final_val = df[col].iloc[-1]
        ax.annotate(f'{col}: {final_val:.1f}',
                    xy=(df.index[-1], final_val),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=10, color=color, fontweight='bold')

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Normalized Price (Base=100)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 对比图已保存: {save_path}")
    plt.close()
    return save_path


if __name__ == "__main__":
    # 从命令行获取参数
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    period = sys.argv[2] if len(sys.argv) > 2 else "2y"

    # 1. 获取单只股票数据
    df = get_stock_data(symbol, period)
    if not df.empty:
        plot_stock_chart(df, symbol)

    # 2. 多股票对比（科技巨头）
    tech_giants = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
    comparison = get_multiple_stocks(tech_giants, "1y")
    if not comparison.empty:
        plot_comparison(comparison)
        print("\n📈 最近1年收益排名:")
        print("-" * 40)
        returns = (comparison.iloc[-1] - 100).sort_values(ascending=False)
        for symbol, ret in returns.items():
            emoji = "🟢" if ret >= 0 else "🔴"
            print(f"  {emoji} {symbol}: {ret:+.2f}%")
