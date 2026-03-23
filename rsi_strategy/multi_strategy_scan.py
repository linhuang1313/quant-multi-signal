"""
多策略 × 多指标 个股全面评分系统
=================================
策略类型:
  1. RSI(2) 均值回归
  2. 布林带均值回归
  3. 动量突破 (N日新高)
  4. 波动率收缩突破 (ATR squeeze)
  5. 均线回踩 (MA pullback)

评估指标:
  - Sharpe
  - Sortino (只惩罚下行波动)
  - Calmar (CAGR / MaxDD)
  - 胜率
  - 盈亏比 (avg win / avg loss)
  - 最大连续亏损笔数
  - 熊市表现 (2020.02-03, 2022全年)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download(ticker):
    df = yf.download(ticker, start='2010-01-01', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=['Close'])

def calc_rsi(series, period=2):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def multi_metrics(trades_list):
    """计算多维度评估指标"""
    if not trades_list or len(trades_list) < 15:
        return None
    tdf = pd.DataFrame(trades_list)
    n = len(tdf)
    rets = tdf['ret'].values / 100
    
    if len(rets) < 2 or np.std(rets) == 0:
        return None
    
    avg_h = tdf['hold_days'].mean()
    tpy = 252 / max(avg_h, 0.5)
    
    # Sharpe
    sharpe = (rets.mean() / rets.std()) * np.sqrt(tpy)
    
    # Sortino (只用下行标准差)
    downside = rets[rets < 0]
    down_std = np.std(downside) if len(downside) > 1 else np.std(rets)
    sortino = (rets.mean() / down_std) * np.sqrt(tpy) if down_std > 0 else 0
    
    # Max Drawdown
    eq = np.cumprod(np.concatenate([[1], 1 + rets]))
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak * 100).min()
    
    # CAGR
    years = max((tdf['exit_date'].iloc[-1] - tdf['entry_date'].iloc[0]).days / 365.25, 0.5)
    total = eq[-1]
    cagr = (total ** (1/years) - 1) * 100
    
    # Calmar
    calmar = cagr / abs(dd) if dd != 0 else 0
    
    # 胜率 & 盈亏比
    wins = tdf[tdf['ret'] > 0]
    losses = tdf[tdf['ret'] < 0]
    win_rate = len(wins) / n * 100
    avg_win = wins['ret'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['ret'].mean()) if len(losses) > 0 else 0.001
    profit_factor = avg_win / avg_loss
    
    # 最大连续亏损
    max_consec_loss = 0
    consec = 0
    for r in rets:
        if r < 0:
            consec += 1
            max_consec_loss = max(max_consec_loss, consec)
        else:
            consec = 0
    
    # 熊市表现 (2020.02-03 COVID, 2022全年)
    bear_trades = tdf[
        ((tdf['entry_date'] >= '2020-02-01') & (tdf['entry_date'] <= '2020-04-30')) |
        ((tdf['entry_date'] >= '2022-01-01') & (tdf['entry_date'] <= '2022-12-31'))
    ]
    bear_avg = bear_trades['ret'].mean() if len(bear_trades) > 0 else np.nan
    
    return {
        'trades': n, 'years': round(years,1), 'trades_yr': round(n/years,1),
        'win_rate': round(win_rate,1), 'avg_ret': round(tdf['ret'].mean(),3),
        'sharpe': round(sharpe,2), 'sortino': round(sortino,2),
        'cagr': round(cagr,2), 'max_dd': round(dd,1),
        'calmar': round(calmar,3),
        'profit_factor': round(profit_factor,2),
        'max_consec_loss': max_consec_loss,
        'bear_avg_ret': round(bear_avg,3) if not np.isnan(bear_avg) else None,
        'worst': round(tdf['ret'].min(),1),
    }

# ============================================================
# 5种策略
# ============================================================
def strat_rsi(df, rsi_th=10, stop=0.05):
    """RSI(2)均值回归"""
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA5'] = df['Close'].rolling(5).mean()
    trades = []
    in_pos = False; ep = ei = 0
    for i in range(201, len(df)):
        c = float(df['Close'].iloc[i]); r2 = float(df['RSI2'].iloc[i])
        s200 = float(df['SMA200'].iloc[i]); s5 = float(df['SMA5'].iloc[i])
        if not in_pos:
            if c > s200 and r2 < rsi_th: ep=c; ei=i; in_pos=True
        else:
            h=i-ei; pnl=(c-ep)/ep
            if c>s5 or pnl<=-stop or h>=10:
                trades.append({'ret':(c-ep)/ep*100,'hold_days':h,'entry_date':df.index[ei],'exit_date':df.index[i]})
                in_pos=False
    return trades

def strat_bollinger(df, period=20, std_mult=2, stop=0.05):
    """布林带均值回归: 价格跌破下轨买入, 回到中轨卖出"""
    df['BB_mid'] = df['Close'].rolling(period).mean()
    df['BB_std'] = df['Close'].rolling(period).std()
    df['BB_lower'] = df['BB_mid'] - std_mult * df['BB_std']
    df['SMA200'] = df['Close'].rolling(200).mean()
    trades = []
    in_pos = False; ep = ei = 0
    for i in range(201, len(df)):
        c = float(df['Close'].iloc[i])
        s200 = float(df['SMA200'].iloc[i])
        mid = float(df['BB_mid'].iloc[i])
        lower = float(df['BB_lower'].iloc[i])
        if not in_pos:
            if c > s200 and c < lower: ep=c; ei=i; in_pos=True
        else:
            h=i-ei; pnl=(c-ep)/ep
            if c>mid or pnl<=-stop or h>=15:
                trades.append({'ret':(c-ep)/ep*100,'hold_days':h,'entry_date':df.index[ei],'exit_date':df.index[i]})
                in_pos=False
    return trades

def strat_momentum(df, lookback=20, stop=0.05):
    """动量突破: N日新高买入, 跌破MA10卖出"""
    df['High_N'] = df['High'].rolling(lookback).max().shift(1)
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA10'] = df['Close'].rolling(10).mean()
    trades = []
    in_pos = False; ep = ei = 0
    for i in range(201, len(df)):
        c = float(df['Close'].iloc[i])
        s200 = float(df['SMA200'].iloc[i])
        s10 = float(df['SMA10'].iloc[i])
        hn = float(df['High_N'].iloc[i]) if not np.isnan(df['High_N'].iloc[i]) else 0
        if not in_pos:
            if c > s200 and c > hn and hn > 0: ep=c; ei=i; in_pos=True
        else:
            h=i-ei; pnl=(c-ep)/ep
            if c<s10 or pnl<=-stop or h>=20:
                trades.append({'ret':(c-ep)/ep*100,'hold_days':h,'entry_date':df.index[ei],'exit_date':df.index[i]})
                in_pos=False
    return trades

def strat_atr_squeeze(df, atr_period=14, squeeze_pct=0.6, stop=0.05):
    """ATR收缩突破: ATR降到近期低位 + 向上突破"""
    df['ATR'] = (df['High']-df['Low']).rolling(atr_period).mean()
    df['ATR_min'] = df['ATR'].rolling(50).min()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['High5'] = df['High'].rolling(5).max().shift(1)
    trades = []
    in_pos = False; ep = ei = 0
    for i in range(201, len(df)):
        c = float(df['Close'].iloc[i])
        s200 = float(df['SMA200'].iloc[i])
        s10 = float(df['SMA10'].iloc[i])
        atr = float(df['ATR'].iloc[i])
        atr_min = float(df['ATR_min'].iloc[i])
        h5 = float(df['High5'].iloc[i]) if not np.isnan(df['High5'].iloc[i]) else c*2
        if not in_pos:
            squeeze = atr < atr_min * (1+squeeze_pct) if atr_min > 0 else False
            if c > s200 and squeeze and c > h5: ep=c; ei=i; in_pos=True
        else:
            h=i-ei; pnl=(c-ep)/ep
            if c<s10 or pnl<=-stop or h>=15:
                trades.append({'ret':(c-ep)/ep*100,'hold_days':h,'entry_date':df.index[ei],'exit_date':df.index[i]})
                in_pos=False
    return trades

def strat_ma_pullback(df, stop=0.05):
    """均线回踩: 强势股回踩MA20买入, 创新高卖出"""
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['High10'] = df['High'].rolling(10).max()
    trades = []
    in_pos = False; ep = ei = 0
    for i in range(201, len(df)):
        c = float(df['Close'].iloc[i])
        s200 = float(df['SMA200'].iloc[i])
        s20 = float(df['SMA20'].iloc[i])
        s50 = float(df['SMA50'].iloc[i])
        h10 = float(df['High10'].iloc[i])
        if not in_pos:
            above_trend = c > s200 and s20 > s50  # 上升趋势
            near_ma20 = abs(c - s20) / s20 < 0.01  # 接近MA20
            if above_trend and near_ma20 and c < s20: ep=c; ei=i; in_pos=True
        else:
            h=i-ei; pnl=(c-ep)/ep
            if c>h10 or pnl<=-stop or h>=15:
                trades.append({'ret':(c-ep)/ep*100,'hold_days':h,'entry_date':df.index[ei],'exit_date':df.index[i]})
                in_pos=False
    return trades

# ============================================================
# 主流程: 对Top标的跑全部5种策略
# ============================================================
# 从之前的结果中取Sharpe排名前80的 + 一些低价股补充
prev = pd.read_csv('/home/user/workspace/quant-trading/rsi_strategy/data/individual_stock_results.csv')
top_tickers = prev.sort_values('sharpe', ascending=False).head(80)['ticker'].tolist()

# 补充一些低价高流动性的
extra = [t for t in prev[prev['price']<100].sort_values('sharpe', ascending=False).head(20)['ticker'].tolist() 
         if t not in top_tickers]
all_tickers = list(dict.fromkeys(top_tickers + extra))

print(f"对 {len(all_tickers)} 只股票运行5种策略...\n")

strategies = {
    'RSI': strat_rsi,
    'Bollinger': strat_bollinger,
    'Momentum': strat_momentum,
    'ATR_Squeeze': strat_atr_squeeze,
    'MA_Pullback': strat_ma_pullback,
}

all_results = []
for idx, ticker in enumerate(all_tickers):
    try:
        df = download(ticker)
        if len(df) < 500:
            continue
        price = round(float(df['Close'].iloc[-1]), 2)
        
        for sname, sfunc in strategies.items():
            trades = sfunc(df.copy())
            metrics = multi_metrics(trades)
            if metrics:
                metrics['ticker'] = ticker
                metrics['strategy'] = sname
                metrics['price'] = price
                all_results.append(metrics)
    except:
        pass
    
    if (idx+1) % 20 == 0:
        print(f"  {idx+1}/{len(all_tickers)} 完成...")

rdf = pd.DataFrame(all_results)
print(f"\n总计 {len(rdf)} 个 (标的×策略) 组合\n")

# 综合评分: 标准化后加权
# Sharpe 30% + Sortino 20% + Calmar 15% + 胜率 15% + 盈亏比 10% + 熊市表现 10%
def composite_score(row):
    score = 0
    score += min(row['sharpe'] / 3.0, 1.0) * 30  # 归一化到0-1
    score += min(row['sortino'] / 4.0, 1.0) * 20
    score += min(row['calmar'] / 0.5, 1.0) * 15
    score += (row['win_rate'] / 100) * 15
    score += min(row['profit_factor'] / 3.0, 1.0) * 10
    if row['bear_avg_ret'] is not None:
        bear_score = min(max(row['bear_avg_ret'] + 2, 0) / 4, 1.0)  # -2%→0, +2%→1
        score += bear_score * 10
    else:
        score += 5  # 无熊市数据给中性分
    return round(score, 1)

rdf['composite'] = rdf.apply(composite_score, axis=1)
rdf = rdf.sort_values('composite', ascending=False)

# 每只股票的最佳策略
best_per_stock = rdf.sort_values('composite', ascending=False).drop_duplicates(subset='ticker')
best_per_stock = best_per_stock.sort_values('composite', ascending=False)

print("="*110)
print(f"🏆 多策略×多指标 综合评分 Top 30")
print("="*110)
print(f"\n{'排名':>3s} {'标的':<7s} {'最佳策略':<12s} {'股价':>7s} {'综合分':>6s} {'Sharpe':>7s} "
      f"{'Sortino':>8s} {'Calmar':>7s} {'胜率':>6s} {'盈亏比':>6s} {'回撤':>7s} {'熊市均收':>8s}")
print("─"*110)
for i, (_, r) in enumerate(best_per_stock.head(30).iterrows(), 1):
    flag = "✅" if r['price'] <= 600 else "💰"
    bear = f"{r['bear_avg_ret']:+.2f}%" if r['bear_avg_ret'] is not None else "  N/A"
    print(f"  {i:>2d} {flag}{r['ticker']:<6s} {r['strategy']:<12s} ${r['price']:>6.0f} "
          f"{r['composite']:>5.1f} {r['sharpe']:>6.2f} {r['sortino']:>7.2f} "
          f"{r['calmar']:>6.3f} {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
          f"{r['max_dd']:>6.1f}% {bear}")

# 按策略类型统计
print(f"\n\n{'='*60}")
print("📊 各策略类型表现统计")
print("="*60)
for sname in strategies.keys():
    sub = rdf[rdf['strategy'] == sname]
    if len(sub) == 0: continue
    good = len(sub[sub['sharpe'] >= 1.0])
    print(f"\n  {sname}:")
    print(f"    有效组合: {len(sub)}  |  Sharpe>=1: {good} ({good/len(sub)*100:.0f}%)")
    print(f"    均Sharpe: {sub['sharpe'].mean():.2f}  |  均Sortino: {sub['sortino'].mean():.2f}")
    print(f"    均胜率: {sub['win_rate'].mean():.1f}%  |  均回撤: {sub['max_dd'].mean():.1f}%")

# 保存
rdf.to_csv('/home/user/workspace/quant-trading/rsi_strategy/data/multi_strategy_results.csv', index=False)
best_per_stock.to_csv('/home/user/workspace/quant-trading/rsi_strategy/data/best_strategy_per_stock.csv', index=False)
print(f"\n\n完整结果已保存")

