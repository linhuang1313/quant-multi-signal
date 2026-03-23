"""
大范围个股RSI均值回归回测
覆盖标普500中市值前100+的个股
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
                    entry_price = close
                    entry_idx = i
                    in_pos = True
            else:
                hold = i - entry_idx
                pnl = (close - entry_price) / entry_price
                sell = close > sma5 or pnl <= -stop_loss or hold >= 10
                if sell:
                    ret = (close - entry_price) / entry_price * 100
                    trades.append({'ret': ret, 'hold_days': hold,
                                   'entry_date': df.index[entry_idx], 'exit_date': df.index[i]})
                    in_pos = False
        
        if len(trades) < 30:
            return None
        
        tdf = pd.DataFrame(trades)
        rets = tdf['ret'].values / 100
        avg_h = tdf['hold_days'].mean()
        tpy = 252 / max(avg_h, 0.5)
        sharpe = (rets.mean() / rets.std()) * np.sqrt(tpy) if rets.std() > 0 else 0
        eq = np.cumprod(np.concatenate([[1], 1 + rets]))
        peak = np.maximum.accumulate(eq)
        dd = ((eq - peak) / peak * 100).min()
        years = max((tdf['exit_date'].iloc[-1] - tdf['entry_date'].iloc[0]).days / 365.25, 0.5)
        current_price = float(df['Close'].iloc[-1])
        
        return {
            'ticker': ticker, 'price': round(current_price, 2),
            'trades': len(tdf), 'trades_yr': round(len(tdf)/years, 1),
            'win_rate': round((tdf['ret']>0).mean()*100, 1),
            'avg_ret': round(tdf['ret'].mean(), 3),
            'sharpe': round(sharpe, 2),
            'max_dd': round(dd, 1),
            'worst': round(tdf['ret'].min(), 1),
        }
    except:
        return None

# 标普500市值前120+的股票
tickers = [
    # Mega Cap
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','BRK-B','LLY','AVGO',
    'JPM','V','UNH','XOM','MA','JNJ','COST','PG','HD','WMT',
    'ABBV','NFLX','CRM','BAC','CVX','MRK','KO','PEP','AMD','TMO',
    'LIN','ORCL','CSCO','ACN','ADBE','MCD','WFC','PM','ABT','IBM',
    'GE','ISRG','CAT','INTU','VZ','AMGN','TXN','NOW','QCOM','GS',
    'PFE','BLK','MS','RTX','SCHW','NEE','AMAT','DHR','T','LOW',
    'HON','UNP','COP','ELV','DE','BA','ADP','TJX','BKNG','MDLZ',
    'SYK','REGN','GILD','VRTX','MMC','LRCX','BSX','ADI','CB','CI',
    'PGR','SBUX','SO','MO','CME','ZTS','ICE','KLAC','DUK','MCK',
    'CMG','PYPL','USB','EOG','SLB','WM','AON','APD','CDNS','SNPS',
    'TGT','EMR','GD','PNC','ITW','HCA','CL','NXPI','SHW','FDX',
    'ABNB','MAR','AJG','CTAS','TFC','PSX','NOC','OXY','SPG','MPC',
    # 一些波动大的热门股
    'COIN','SQ','ROKU','SNOW','CRWD','DDOG','NET','PLTR','SOFI','RBLX',
    'NKE','DIS','F','GM','INTC','UBER','LYFT','SNAP','PINS','SPOT',
]

print("正在回测", len(tickers), "只个股...\n")

results = []
for i, t in enumerate(tickers):
    r = backtest(t)
    if r:
        results.append(r)
    if (i+1) % 20 == 0:
        print(f"  已完成 {i+1}/{len(tickers)}...")

print(f"\n有效结果: {len(results)} 只\n")

rdf = pd.DataFrame(results).sort_values('sharpe', ascending=False)

# Top 20
print("="*95)
print("🏆 RSI(2)均值回归 Sharpe 排名 Top 20 (2010-2026)")
print("="*95)
print(f"\n{'排名':>3s} {'标的':<7s} {'股价':>8s} {'交易数':>5s} {'年均':>5s} {'胜率':>6s} "
      f"{'均收':>7s} {'Sharpe':>7s} {'回撤':>7s} {'最差':>6s}")
print("─"*90)
for i, (_, r) in enumerate(rdf.head(20).iterrows(), 1):
    affordable = "✅" if r['price'] <= 600 else "⚠️"
    print(f"  {i:>2d} {affordable}{r['ticker']:<6s} ${r['price']:>7.2f} {r['trades']:>5d} "
          f"{r['trades_yr']:>5.1f} {r['win_rate']:>5.1f}% {r['avg_ret']:>+6.3f}% "
          f"{r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% {r['worst']:>5.1f}%")

# Sharpe >= 1.5 的完整列表
good = rdf[rdf['sharpe'] >= 1.5]
print(f"\n\n{'='*95}")
print(f"📊 Sharpe >= 1.5 的个股完整列表 ({len(good)}只)")
print("="*95)
print(f"\n{'标的':<7s} {'股价':>8s} {'交易':>5s} {'年均':>5s} {'胜率':>6s} "
      f"{'均收':>7s} {'Sharpe':>7s} {'回撤':>7s}")
print("─"*60)
for _, r in good.iterrows():
    affordable = "✅" if r['price'] <= 600 else "💰"
    print(f"{affordable}{r['ticker']:<6s} ${r['price']:>7.2f} {r['trades']:>5d} "
          f"{r['trades_yr']:>5.1f} {r['win_rate']:>5.1f}% {r['avg_ret']:>+6.3f}% "
          f"{r['sharpe']:>6.2f} {r['max_dd']:>6.1f}%")

# 统计
print(f"\n  Sharpe >= 2.0: {len(rdf[rdf['sharpe']>=2.0])} 只")
print(f"  Sharpe >= 1.5: {len(rdf[rdf['sharpe']>=1.5])} 只")
print(f"  Sharpe >= 1.0: {len(rdf[rdf['sharpe']>=1.0])} 只")
print(f"  Sharpe < 0 (亏损): {len(rdf[rdf['sharpe']<0])} 只")

# 保存完整结果
rdf.to_csv('/home/user/workspace/quant-trading/rsi_strategy/data/individual_stock_results.csv', index=False)
print(f"\n完整结果已保存到 data/individual_stock_results.csv")

