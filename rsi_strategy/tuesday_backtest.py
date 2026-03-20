"""
Turnaround Tuesday 策略 — 独立回测
====================================
周一大跌后周二大概率反弹

测试多种变体:
1. 基础版: 周一跌1% → 周二卖
2. IBS过滤版: 周一收阴 + IBS<0.2 → 周二卖
3. 延长持有版: 周一跌 + IBS<0.5 → 持有到价格>昨日高点 或 4天后
4. 综合版: 周一跌 + IBS<0.5 → 周二卖
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download_data(ticker, start='1993-01-01'):
    df = yf.download(ticker, start=start, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def backtest_tuesday(df, strategy_name, entry_func, exit_func, commission=0.0003):
    """回测引擎 — 按日逐行扫描"""
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None
    entry_idx = 0
    
    for i in range(2, len(df)):
        if not in_position:
            if entry_func(df, i):
                entry_price = df['Close'].iloc[i]
                entry_date = df.index[i]
                entry_idx = i
                in_position = True
        else:
            if exit_func(df, i, entry_idx):
                exit_price = df['Close'].iloc[i]
                exit_date = df.index[i]
                hold_days = i - entry_idx
                
                gross_ret = (exit_price - entry_price) / entry_price
                net_ret = gross_ret - 2 * commission
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'hold_days': hold_days,
                    'return_pct': float(net_ret * 100),
                })
                in_position = False
    
    if not trades:
        return {'strategy': strategy_name, 'total_trades': 0}
    
    tdf = pd.DataFrame(trades)
    total = len(tdf)
    wr = (tdf['return_pct'] > 0).mean() * 100
    avg_ret = tdf['return_pct'].mean()
    med_ret = tdf['return_pct'].median()
    avg_win = tdf[tdf['return_pct'] > 0]['return_pct'].mean() if (tdf['return_pct'] > 0).any() else 0
    avg_loss = tdf[tdf['return_pct'] < 0]['return_pct'].mean() if (tdf['return_pct'] < 0).any() else 0
    
    cum = (1 + tdf['return_pct']/100).prod()
    total_ret = (cum - 1) * 100
    years = (tdf['exit_date'].iloc[-1] - tdf['entry_date'].iloc[0]).days / 365.25
    cagr = (cum ** (1/years) - 1) * 100 if years > 0 else 0
    
    total_hold = tdf['hold_days'].sum()
    total_cal = (tdf['exit_date'].iloc[-1] - tdf['entry_date'].iloc[0]).days
    exposure = total_hold / (total_cal * 5/7) * 100 if total_cal > 0 else 0
    
    eq = [1.0]
    for r in tdf['return_pct']:
        eq.append(eq[-1] * (1 + r/100))
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()
    
    rets = tdf['return_pct'].values / 100
    if len(rets) > 1 and rets.std() > 0:
        avg_h = tdf['hold_days'].mean()
        tpy = 252 / avg_h if avg_h > 0 else 50
        sharpe = (rets.mean() / rets.std()) * np.sqrt(tpy)
    else:
        sharpe = 0
    
    gp = tdf[tdf['return_pct'] > 0]['return_pct'].sum()
    gl = abs(tdf[tdf['return_pct'] < 0]['return_pct'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    
    return {
        'strategy': strategy_name,
        'total_trades': total,
        'win_rate': round(wr, 1),
        'avg_return': round(avg_ret, 3),
        'median_return': round(med_ret, 3),
        'avg_winner': round(avg_win, 3),
        'avg_loser': round(avg_loss, 3),
        'total_return': round(total_ret, 1),
        'cagr': round(cagr, 2),
        'exposure': round(exposure, 1),
        'max_drawdown': round(max_dd, 1),
        'sharpe': round(sharpe, 2),
        'profit_factor': round(pf, 2),
        'avg_hold_days': round(tdf['hold_days'].mean(), 1),
        'years': round(years, 1),
    }

def run_tuesday(ticker='SPY'):
    print(f"\n{'='*75}")
    print(f"📊 Turnaround Tuesday 回测 — {ticker}")
    print(f"{'='*75}")
    
    df = download_data(ticker)
    print(f"  数据: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({len(df)}交易日)")
    
    # 预计算
    df['weekday'] = df.index.weekday  # 0=Mon
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['prev_close'] = df['Close'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['friday_close'] = df['Close'].shift(1)  # 简化：前一个交易日的收盘
    
    results = []
    
    # ── 策略1: 基础版 — 周一跌1% → 周二收盘卖 ──
    def entry1(df, i):
        return (df['weekday'].iloc[i] == 0 and  # 周一
                df['Close'].iloc[i] < df['prev_close'].iloc[i] * 0.99)  # 跌1%
    def exit1(df, i, ei):
        return i >= ei + 1  # 下一个交易日（周二）收盘
    results.append(backtest_tuesday(df, "策略1: 周一跌1% → 周二卖", entry1, exit1))
    
    # ── 策略2: IBS过滤 — 周一收阴 + IBS<0.2 → 周二卖 ──
    def entry2(df, i):
        return (df['weekday'].iloc[i] == 0 and
                df['Close'].iloc[i] < df['Open'].iloc[i] and  # 收阴
                df['IBS'].iloc[i] < 0.2)
    results.append(backtest_tuesday(df, "策略2: 周一收阴+IBS<0.2 → 周二卖", entry2, exit1))
    
    # ── 策略3: IBS<0.5 + 周一跌 → 周二卖 ──
    def entry3(df, i):
        return (df['weekday'].iloc[i] == 0 and
                df['Close'].iloc[i] < df['prev_close'].iloc[i] and
                df['IBS'].iloc[i] < 0.5)
    results.append(backtest_tuesday(df, "策略3: 周一跌+IBS<0.5 → 周二卖", entry3, exit1))
    
    # ── 策略4: 延长持有 — 周一跌+IBS<0.5 → 价格>昨高 或 4天后 ──
    def entry4(df, i):
        return (df['weekday'].iloc[i] == 0 and
                df['Close'].iloc[i] < df['prev_close'].iloc[i] and
                df['IBS'].iloc[i] < 0.5)
    def exit4(df, i, ei):
        if i - ei >= 4:  # 最多持4天
            return True
        if i > ei and df['Close'].iloc[i] > df['High'].iloc[i-1]:  # 收盘>昨日高点
            return True
        return False
    results.append(backtest_tuesday(df, "策略4: 周一跌+IBS<0.5 → >昨高或4天", entry4, exit4))
    
    # ── 策略5: 大跌版 — 周一跌>1.5% + IBS<0.3 → 周二卖 ──
    def entry5(df, i):
        return (df['weekday'].iloc[i] == 0 and
                df['Close'].iloc[i] < df['prev_close'].iloc[i] * 0.985 and
                df['IBS'].iloc[i] < 0.3)
    results.append(backtest_tuesday(df, "策略5: 周一跌1.5%+IBS<0.3 → 周二卖", entry5, exit1))
    
    # ── 策略6: 综合最优 — 周一跌+IBS<0.3 → 持有到>昨高或3天 ──
    def entry6(df, i):
        return (df['weekday'].iloc[i] == 0 and
                df['Close'].iloc[i] < df['prev_close'].iloc[i] and
                df['IBS'].iloc[i] < 0.3)
    def exit6(df, i, ei):
        if i - ei >= 3:
            return True
        if i > ei and df['Close'].iloc[i] > df['High'].iloc[i-1]:
            return True
        return False
    results.append(backtest_tuesday(df, "策略6: 周一跌+IBS<0.3 → >昨高或3天", entry6, exit6))
    
    # 输出
    print(f"\n{'─'*75}")
    print(f"{'策略':<40s} {'交易':>5s} {'胜率':>6s} {'均回报':>7s} {'CAGR':>7s} "
          f"{'回撤':>6s} {'Sharpe':>7s} {'PF':>5s} {'暴露':>5s}")
    print(f"{'─'*75}")
    
    for r in results:
        if r['total_trades'] == 0:
            print(f"  {r['strategy']:<38s}  无交易")
            continue
        print(f"  {r['strategy']:<38s} {r['total_trades']:>5d} {r['win_rate']:>5.1f}% "
              f"{r['avg_return']:>+6.3f}% {r['cagr']:>6.2f}% "
              f"{r['max_drawdown']:>5.1f}% {r['sharpe']:>6.2f} {r['profit_factor']:>5.2f} "
              f"{r['exposure']:>4.1f}%")
    
    print(f"{'─'*75}")
    
    best = max(results, key=lambda x: x.get('sharpe', 0))
    print(f"\n🏆 最佳: {best['strategy']}")
    print(f"   {best['total_trades']}笔 | 胜率{best['win_rate']}% | "
          f"每笔{best['avg_return']:+.3f}% | CAGR {best['cagr']}% | "
          f"回撤{best['max_drawdown']}% | Sharpe {best['sharpe']} | "
          f"暴露{best['exposure']}%")
    
    return results

if __name__ == '__main__':
    spy = run_tuesday('SPY')
    print("\n")
    qqq = run_tuesday('QQQ')
