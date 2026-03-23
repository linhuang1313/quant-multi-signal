"""
大范围个股回测 Part 2 — 新增375只
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calc_rsi(series, period=2):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def backtest(ticker, rsi_entry=10, stop_loss=0.05):
    try:
        df = yf.download(ticker, start='2010-01-01', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=['Close'])
        if len(df) < 500:
            return None
        df['RSI2'] = calc_rsi(df['Close'], 2)
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['SMA5'] = df['Close'].rolling(5).mean()
        
        trades = []
        in_pos = False
        entry_price = entry_idx = 0
        for i in range(201, len(df)):
            close = float(df['Close'].iloc[i])
            rsi2 = float(df['RSI2'].iloc[i])
            sma200 = float(df['SMA200'].iloc[i])
            sma5 = float(df['SMA5'].iloc[i])
            if not in_pos:
                if close > sma200 and rsi2 < rsi_entry:
                    entry_price = close; entry_idx = i; in_pos = True
            else:
                hold = i - entry_idx
                pnl = (close - entry_price) / entry_price
                if close > sma5 or pnl <= -stop_loss or hold >= 10:
                    trades.append({'ret': (close-entry_price)/entry_price*100, 'hold_days': hold,
                                   'entry_date': df.index[entry_idx], 'exit_date': df.index[i]})
                    in_pos = False
        if len(trades) < 30: return None
        tdf = pd.DataFrame(trades)
        rets = tdf['ret'].values / 100
        avg_h = tdf['hold_days'].mean()
        tpy = 252 / max(avg_h, 0.5)
        sharpe = (rets.mean()/rets.std())*np.sqrt(tpy) if rets.std()>0 else 0
        eq = np.cumprod(np.concatenate([[1], 1+rets]))
        dd = ((eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq)*100).min()
        years = max((tdf['exit_date'].iloc[-1]-tdf['entry_date'].iloc[0]).days/365.25, 0.5)
        return {
            'ticker': ticker, 'price': round(float(df['Close'].iloc[-1]),2),
            'trades': len(tdf), 'trades_yr': round(len(tdf)/years,1),
            'win_rate': round((tdf['ret']>0).mean()*100,1),
            'avg_ret': round(tdf['ret'].mean(),3),
            'sharpe': round(sharpe,2), 'max_dd': round(dd,1),
            'worst': round(tdf['ret'].min(),1),
        }
    except: return None

# 加载新tickers
with open('/home/user/workspace/quant-trading/rsi_strategy/data/new_tickers.txt') as f:
    tickers = [t.strip() for t in f if t.strip()]

print(f"回测 {len(tickers)} 只新股票...\n")
results = []
for i, t in enumerate(tickers):
    r = backtest(t)
    if r: results.append(r)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(tickers)} 完成... (有效 {len(results)})")

print(f"\n新增有效: {len(results)} 只")

# 合并之前的结果
old = pd.read_csv('/home/user/workspace/quant-trading/rsi_strategy/data/individual_stock_results.csv')
new = pd.DataFrame(results)
combined = pd.concat([old, new], ignore_index=True).drop_duplicates(subset='ticker')
combined = combined.sort_values('sharpe', ascending=False)
combined.to_csv('/home/user/workspace/quant-trading/rsi_strategy/data/individual_stock_results.csv', index=False)

print(f"总计有效: {len(combined)} 只\n")

# 最终统计
print("="*95)
print(f"🏆 全美股RSI(2)均值回归 — {len(combined)} 只个股回测结果")
print("="*95)
print(f"\n  Sharpe >= 2.0: {len(combined[combined['sharpe']>=2.0])} 只")
print(f"  Sharpe >= 1.5: {len(combined[combined['sharpe']>=1.5])} 只")
print(f"  Sharpe >= 1.0: {len(combined[combined['sharpe']>=1.0])} 只")
print(f"  Sharpe 0-1.0:  {len(combined[(combined['sharpe']>=0)&(combined['sharpe']<1)])} 只")
print(f"  Sharpe < 0:    {len(combined[combined['sharpe']<0])} 只")

# Top 30
top30 = combined.head(30)
print(f"\n\nTop 30 (Sharpe排序):")
print(f"{'排名':>3s} {'标的':<7s} {'股价':>8s} {'交易':>5s} {'年均':>5s} {'胜率':>6s} "
      f"{'均收':>7s} {'Sharpe':>7s} {'回撤':>7s}")
print("─"*65)
for i, (_, r) in enumerate(top30.iterrows(), 1):
    flag = "✅" if r['price'] <= 600 else "💰"
    print(f"  {i:>2d} {flag}{r['ticker']:<6s} ${r['price']:>7.2f} {r['trades']:>5.0f} "
          f"{r['trades_yr']:>5.1f} {r['win_rate']:>5.1f}% {r['avg_ret']:>+6.3f}% "
          f"{r['sharpe']:>6.2f} {r['max_dd']:>6.1f}%")

