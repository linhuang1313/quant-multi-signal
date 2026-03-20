"""
RSI均值回归策略 — 独立回测引擎
=================================
基于 Larry Connors RSI(2) 策略及多种变体
在 SPY / QQQ 上做完整回测验证
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据下载
# ============================================================
def download_data(ticker, start='1993-01-01'):
    """下载历史日线数据"""
    df = yf.download(ticker, start=start, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df

# ============================================================
# 2. 技术指标
# ============================================================
def calc_rsi(series, period=2):
    """计算RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_sma(series, period):
    return series.rolling(period).mean()

def calc_ibs(df):
    """Internal Bar Strength = (Close - Low) / (High - Low)"""
    return (df['Close'] - df['Low']) / (df['High'] - df['Low'])

# ============================================================
# 3. 回测引擎
# ============================================================
def backtest_strategy(df, strategy_name, entry_func, exit_func, commission=0.0003):
    """
    通用回测引擎
    
    Args:
        df: OHLCV DataFrame
        strategy_name: 策略名称
        entry_func: 入场条件函数 f(df, i) -> bool
        exit_func: 出场条件函数 f(df, i) -> bool
        commission: 单边手续费率
    
    Returns:
        dict: 回测结果
    """
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None
    entry_idx = 0
    
    for i in range(1, len(df)):
        if not in_position:
            if entry_func(df, i):
                entry_price = df['Close'].iloc[i]
                entry_date = df.index[i]
                entry_idx = i
                in_position = True
        else:
            if exit_func(df, i):
                exit_price = df['Close'].iloc[i]
                exit_date = df.index[i]
                hold_days = i - entry_idx
                
                gross_return = (exit_price - entry_price) / entry_price
                net_return = gross_return - 2 * commission  # 双边手续费
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'hold_days': hold_days,
                    'return_pct': net_return * 100,
                    'gross_return': gross_return * 100,
                })
                in_position = False
    
    if not trades:
        return {'strategy': strategy_name, 'total_trades': 0}
    
    trades_df = pd.DataFrame(trades)
    
    # 计算指标
    total_trades = len(trades_df)
    avg_return = trades_df['return_pct'].mean()
    median_return = trades_df['return_pct'].median()
    win_rate = (trades_df['return_pct'] > 0).mean() * 100
    avg_winner = trades_df[trades_df['return_pct'] > 0]['return_pct'].mean() if (trades_df['return_pct'] > 0).any() else 0
    avg_loser = trades_df[trades_df['return_pct'] < 0]['return_pct'].mean() if (trades_df['return_pct'] < 0).any() else 0
    
    # 累计收益 (复利)
    cumulative = (1 + trades_df['return_pct']/100).prod()
    total_return = (cumulative - 1) * 100
    
    # 年化
    first_date = trades_df['entry_date'].iloc[0]
    last_date = trades_df['exit_date'].iloc[-1]
    years = (last_date - first_date).days / 365.25
    cagr = (cumulative ** (1/years) - 1) * 100 if years > 0 else 0
    
    # 持仓时间占比
    total_hold = trades_df['hold_days'].sum()
    total_calendar = (last_date - first_date).days
    exposure = total_hold / (total_calendar * 5/7) * 100 if total_calendar > 0 else 0  # 近似交易日
    
    # 最大回撤 (基于逐笔)
    equity = [1.0]
    for r in trades_df['return_pct']:
        equity.append(equity[-1] * (1 + r/100))
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    # Sharpe (年化)
    returns_arr = trades_df['return_pct'].values / 100
    if len(returns_arr) > 1 and returns_arr.std() > 0:
        avg_hold = trades_df['hold_days'].mean()
        trades_per_year = 252 / avg_hold if avg_hold > 0 else 50
        sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0
    
    # Profit Factor
    gross_profit = trades_df[trades_df['return_pct'] > 0]['return_pct'].sum()
    gross_loss = abs(trades_df[trades_df['return_pct'] < 0]['return_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'strategy': strategy_name,
        'total_trades': total_trades,
        'win_rate': round(win_rate, 1),
        'avg_return': round(avg_return, 3),
        'median_return': round(median_return, 3),
        'avg_winner': round(avg_winner, 3),
        'avg_loser': round(avg_loser, 3),
        'total_return': round(total_return, 1),
        'cagr': round(cagr, 2),
        'exposure': round(exposure, 1),
        'max_drawdown': round(max_drawdown, 1),
        'sharpe': round(sharpe, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_hold_days': round(trades_df['hold_days'].mean(), 1),
        'years': round(years, 1),
        'trades_df': trades_df,
    }

# ============================================================
# 4. 买入持有基准
# ============================================================
def buy_and_hold(df):
    """计算买入持有基准"""
    first = df['Close'].iloc[0]
    last = df['Close'].iloc[-1]
    total_return = (last / first - 1) * 100
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((last/first) ** (1/years) - 1) * 100
    
    # 最大回撤
    equity = df['Close'] / df['Close'].iloc[0]
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()
    
    # 年化Sharpe
    daily_ret = df['Close'].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    
    return {
        'strategy': '买入持有 (基准)',
        'total_return': round(float(total_return), 1),
        'cagr': round(float(cagr), 2),
        'max_drawdown': round(float(max_dd), 1),
        'sharpe': round(float(sharpe), 2),
        'exposure': 100.0,
    }

# ============================================================
# 5. 策略定义
# ============================================================
def run_all(ticker='SPY'):
    print(f"\n{'='*75}")
    print(f"📊 RSI均值回归策略独立回测 — {ticker}")
    print(f"{'='*75}")
    
    # 下载数据
    print(f"\n⏳ 下载 {ticker} 历史数据...")
    df = download_data(ticker)
    print(f"  数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  总交易日: {len(df)}")
    
    # 预计算指标
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['RSI14'] = calc_rsi(df['Close'], 14)
    df['SMA5'] = calc_sma(df['Close'], 5)
    df['SMA10'] = calc_sma(df['Close'], 10)
    df['SMA200'] = calc_sma(df['Close'], 200)
    df['IBS'] = calc_ibs(df)
    df['RSI_long'] = calc_rsi(df['Close'], 30)
    
    # 买入持有基准
    bh = buy_and_hold(df)
    
    results = []
    
    # ─────────────────────────────────────────────
    # 策略1: 基础 RSI(2) < 10, 卖出 RSI(2) > 90
    # ─────────────────────────────────────────────
    def entry_basic(df, i):
        return df['RSI2'].iloc[i] < 10
    def exit_basic(df, i):
        return df['RSI2'].iloc[i] > 90
    
    results.append(backtest_strategy(df, "策略1: RSI(2)<10 → RSI(2)>90", entry_basic, exit_basic))
    
    # ─────────────────────────────────────────────
    # 策略2: Connors经典 — 价格>MA200 + RSI(2)<5 → 价格>MA5
    # ─────────────────────────────────────────────
    def entry_connors(df, i):
        return (df['Close'].iloc[i] > df['SMA200'].iloc[i] and 
                df['RSI2'].iloc[i] < 5)
    def exit_connors(df, i):
        return df['Close'].iloc[i] > df['SMA5'].iloc[i]
    
    results.append(backtest_strategy(df, "策略2: Connors经典 (>MA200 + RSI2<5 → >MA5)", entry_connors, exit_connors))
    
    # ─────────────────────────────────────────────
    # 策略3: Connors + RSI(2)<15更宽松入场
    # ─────────────────────────────────────────────
    def entry_connors15(df, i):
        return (df['Close'].iloc[i] > df['SMA200'].iloc[i] and 
                df['RSI2'].iloc[i] < 15)
    
    results.append(backtest_strategy(df, "策略3: >MA200 + RSI2<15 → >MA5", entry_connors15, exit_connors))
    
    # ─────────────────────────────────────────────
    # 策略4: 双RSI — 长期RSI>50 + 短期RSI<15 → RSI>85
    # ─────────────────────────────────────────────
    def entry_dual(df, i):
        return (df['RSI_long'].iloc[i] > 50 and 
                df['RSI2'].iloc[i] < 15)
    def exit_dual(df, i):
        return df['RSI2'].iloc[i] > 85
    
    results.append(backtest_strategy(df, "策略4: 双RSI (RSI30>50 + RSI2<15 → RSI2>85)", entry_dual, exit_dual))
    
    # ─────────────────────────────────────────────
    # 策略5: Connors + IBS过滤 (更低买入价)
    # ─────────────────────────────────────────────
    def entry_ibs(df, i):
        return (df['Close'].iloc[i] > df['SMA200'].iloc[i] and 
                df['RSI2'].iloc[i] < 10 and
                df['IBS'].iloc[i] < 0.3)
    
    results.append(backtest_strategy(df, "策略5: >MA200 + RSI2<10 + IBS<0.3 → >MA5", entry_ibs, exit_connors))
    
    # ─────────────────────────────────────────────
    # 策略6: 连续下跌 + RSI — 连跌2天 + RSI<15
    # ─────────────────────────────────────────────
    def entry_consec(df, i):
        if i < 2:
            return False
        consec_down = (df['Close'].iloc[i] < df['Close'].iloc[i-1] and 
                       df['Close'].iloc[i-1] < df['Close'].iloc[i-2])
        return (df['Close'].iloc[i] > df['SMA200'].iloc[i] and 
                consec_down and
                df['RSI2'].iloc[i] < 15)
    
    results.append(backtest_strategy(df, "策略6: >MA200 + 连跌2天 + RSI2<15 → >MA5", entry_consec, exit_connors))
    
    # ─────────────────────────────────────────────
    # 策略7: 累积RSI — 连续3天RSI2之和 < 20
    # ─────────────────────────────────────────────
    def entry_cumrsi(df, i):
        if i < 3:
            return False
        cum_rsi = df['RSI2'].iloc[i] + df['RSI2'].iloc[i-1] + df['RSI2'].iloc[i-2]
        return (df['Close'].iloc[i] > df['SMA200'].iloc[i] and 
                cum_rsi < 20)
    
    results.append(backtest_strategy(df, "策略7: >MA200 + 3日累积RSI2<20 → >MA5", entry_cumrsi, exit_connors))
    
    # ─────────────────────────────────────────────
    # 输出结果
    # ─────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"📋 回测结果汇总 — {ticker}")
    print(f"{'='*75}")
    
    # 基准
    print(f"\n📌 基准: 买入持有 {ticker}")
    print(f"   总收益: {bh['total_return']:+.1f}%  CAGR: {bh['cagr']:.2f}%  "
          f"最大回撤: {bh['max_drawdown']:.1f}%  Sharpe: {bh['sharpe']:.2f}")
    
    print(f"\n{'─'*75}")
    print(f"{'策略':<45s} {'交易':>5s} {'胜率':>6s} {'均回报':>7s} {'CAGR':>7s} "
          f"{'回撤':>6s} {'Sharpe':>7s} {'PF':>5s} {'暴露':>5s}")
    print(f"{'─'*75}")
    
    for r in results:
        if r['total_trades'] == 0:
            print(f"  {r['strategy']:<43s}  无交易")
            continue
        print(f"  {r['strategy']:<43s} {r['total_trades']:>5d} {r['win_rate']:>5.1f}% "
              f"{r['avg_return']:>+6.2f}% {r['cagr']:>6.2f}% "
              f"{r['max_drawdown']:>5.1f}% {r['sharpe']:>6.2f} {r['profit_factor']:>5.2f} "
              f"{r['exposure']:>4.1f}%")
    
    print(f"{'─'*75}")
    
    # 详细对比最佳策略
    best = max(results, key=lambda x: x.get('sharpe', 0))
    print(f"\n🏆 最佳风险调整收益: {best['strategy']}")
    print(f"   总交易: {best['total_trades']}笔 ({best['years']}年)")
    print(f"   胜率: {best['win_rate']}%")
    print(f"   平均每笔: {best['avg_return']:+.3f}% (赢{best['avg_winner']:+.2f}% / 亏{best['avg_loser']:+.2f}%)")
    print(f"   总收益: {best['total_return']:+.1f}%  CAGR: {best['cagr']}%")
    print(f"   最大回撤: {best['max_drawdown']}%")
    print(f"   Sharpe: {best['sharpe']}")
    print(f"   Profit Factor: {best['profit_factor']}")
    print(f"   平均持仓: {best['avg_hold_days']}天")
    print(f"   市场暴露: {best['exposure']}%")
    
    # 对比买入持有
    print(f"\n📊 vs 买入持有:")
    print(f"   CAGR: {best['cagr']}% vs {bh['cagr']}% ({'优' if best['cagr'] > bh['cagr'] else '劣'})")
    print(f"   回撤: {best['max_drawdown']}% vs {bh['max_drawdown']}% ({'优' if best['max_drawdown'] > bh['max_drawdown'] else '劣'})")
    print(f"   Sharpe: {best['sharpe']} vs {bh['sharpe']} ({'优' if best['sharpe'] > bh['sharpe'] else '劣'})")
    if best['exposure'] < 50:
        print(f"   仅 {best['exposure']}% 时间在场 → 剩余资金可用于其他策略")
    
    return results, bh

# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    # SPY 回测
    spy_results, spy_bh = run_all('SPY')
    
    print("\n\n")
    
    # QQQ 回测
    qqq_results, qqq_bh = run_all('QQQ')
    
    # 总结
    print(f"\n\n{'='*75}")
    print(f"📊 最终总结")
    print(f"{'='*75}")
    
    spy_best = max(spy_results, key=lambda x: x.get('sharpe', 0))
    qqq_best = max(qqq_results, key=lambda x: x.get('sharpe', 0))
    
    print(f"\n  SPY最优: {spy_best['strategy']}")
    print(f"    Sharpe {spy_best['sharpe']} | 胜率 {spy_best['win_rate']}% | CAGR {spy_best['cagr']}% | 回撤 {spy_best['max_drawdown']}%")
    
    print(f"\n  QQQ最优: {qqq_best['strategy']}")
    print(f"    Sharpe {qqq_best['sharpe']} | 胜率 {qqq_best['win_rate']}% | CAGR {qqq_best['cagr']}% | 回撤 {qqq_best['max_drawdown']}%")

