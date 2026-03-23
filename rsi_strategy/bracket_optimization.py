"""
分析历史交易数据，找出最优止损止盈比例
用实际回测数据说话，不拍脑袋
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download(ticker, start='2005-01-01'):
    df = yf.download(ticker, start=start, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
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

# ============================================================
# Part 1: RSI策略的历史交易 — 每笔交易的最大盈利和最大亏损
# ============================================================
def analyze_trades(ticker, rsi_entry, use_ibs=False, ibs_thresh=0.3):
    df = download(ticker)
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    trades = []
    in_pos = False
    entry_price = entry_idx = 0
    
    for i in range(201, len(df)):
        close = float(df['Close'].iloc[i])
        rsi2 = float(df['RSI2'].iloc[i])
        sma200 = float(df['SMA200'].iloc[i])
        sma5 = float(df['SMA5'].iloc[i])
        ibs = float(df['IBS'].iloc[i])
        
        if not in_pos:
            cond = close > sma200 and rsi2 < rsi_entry
            if use_ibs:
                cond = cond and ibs < ibs_thresh
            if cond:
                entry_price = close
                entry_idx = i
                in_pos = True
        else:
            hold = i - entry_idx
            pnl_pct = (close - entry_price) / entry_price * 100
            
            # 正常出场: 价格>MA5 或 持仓>=10天
            sell = close > sma5 or hold >= 10
            
            if sell:
                # 记录这笔交易期间的最大盈利和最大亏损(盘中)
                trade_slice = df.iloc[entry_idx+1:i+1]
                max_high = float(trade_slice['High'].max())
                min_low = float(trade_slice['Low'].min())
                max_gain = (max_high - entry_price) / entry_price * 100
                max_loss = (min_low - entry_price) / entry_price * 100
                
                trades.append({
                    'entry_date': df.index[entry_idx],
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': close,
                    'pnl_pct': pnl_pct,
                    'max_gain_pct': max_gain,  # 持仓期间盘中最大浮盈
                    'max_loss_pct': max_loss,  # 持仓期间盘中最大浮亏
                    'hold_days': hold,
                })
                in_pos = False
    
    return pd.DataFrame(trades)

# 分析三个标的
for ticker, rsi_entry, use_ibs in [('SPY', 10, True), ('QQQ', 15, False), ('GLD', 10, False)]:
    tdf = analyze_trades(ticker, rsi_entry, use_ibs)
    if tdf.empty:
        continue
    
    print(f"\n{'='*70}")
    print(f"📊 {ticker} RSI策略 — {len(tdf)} 笔历史交易分析")
    print(f"{'='*70}")
    
    # 盈亏分布
    print(f"\n  最终盈亏分布:")
    print(f"    平均: {tdf['pnl_pct'].mean():+.2f}%")
    print(f"    中位: {tdf['pnl_pct'].median():+.2f}%")
    print(f"    胜率: {(tdf['pnl_pct']>0).mean()*100:.1f}%")
    
    # 盘中最大浮盈分布
    print(f"\n  盘中最大浮盈分布 (持仓期间High):")
    for p in [25, 50, 75, 90, 95]:
        print(f"    {p}th: +{tdf['max_gain_pct'].quantile(p/100):.2f}%")
    
    # 盘中最大浮亏分布
    print(f"\n  盘中最大浮亏分布 (持仓期间Low):")
    for p in [5, 10, 25, 50]:
        print(f"    {p}th: {tdf['max_loss_pct'].quantile(p/100):.2f}%")
    
    # 关键问题：止盈设多少？
    # 如果设了止盈X%，有多少交易会被提前止盈而不是等MA5出场？
    print(f"\n  止盈分析 — 设了止盈会提前截断多少盈利:")
    for tp in [1, 2, 3, 4, 5, 6, 8, 10]:
        # 在盘中最大浮盈 >= tp% 的交易中，有多少最终收益更高？
        would_trigger = (tdf['max_gain_pct'] >= tp).sum()
        pct_trigger = would_trigger / len(tdf) * 100
        
        # 触发止盈的交易，最终平均收益是多少？
        triggered = tdf[tdf['max_gain_pct'] >= tp]
        if len(triggered) > 0:
            avg_final = triggered['pnl_pct'].mean()
            # 如果止盈了，收益就是tp%；如果没止盈，收益不变
            not_triggered = tdf[tdf['max_gain_pct'] < tp]
            simulated_avg = (len(triggered) * tp + not_triggered['pnl_pct'].sum()) / len(tdf)
        else:
            avg_final = 0
            simulated_avg = tdf['pnl_pct'].mean()
        
        print(f"    止盈 {tp:>2d}%: {pct_trigger:>5.1f}% 交易触发, "
              f"原始均收 {tdf['pnl_pct'].mean():+.3f}% → 止盈后 {simulated_avg:+.3f}%")
    
    # 止损分析
    print(f"\n  止损分析 — 止损被触发的比例和影响:")
    for sl in [2, 3, 4, 5, 7, 10]:
        would_stop = (tdf['max_loss_pct'] <= -sl).sum()
        pct_stop = would_stop / len(tdf) * 100
        
        stopped = tdf[tdf['max_loss_pct'] <= -sl]
        not_stopped = tdf[tdf['max_loss_pct'] > -sl]
        if len(stopped) > 0:
            # 被止损的交易最终收益（如果不止损的话）
            avg_stopped_final = stopped['pnl_pct'].mean()
            simulated_avg = (len(stopped) * (-sl) + not_stopped['pnl_pct'].sum()) / len(tdf)
        else:
            avg_stopped_final = 0
            simulated_avg = tdf['pnl_pct'].mean()
        
        print(f"    止损-{sl:>2d}%: {pct_stop:>5.1f}% 交易触发, "
              f"被止损交易原均收 {avg_stopped_final:+.3f}% → 止损后组合均收 {simulated_avg:+.3f}%")

# ============================================================
# Part 2: Tuesday策略
# ============================================================
print(f"\n\n{'='*70}")
print(f"📊 Tuesday策略 — 历史交易分析")
print(f"{'='*70}")

spy = download('SPY')
spy['prev_close'] = spy['Close'].shift(1)
spy['day_ret'] = (spy['Close'] - spy['prev_close']) / spy['prev_close']

trades_tue = []
for i in range(1, len(spy)-1):
    if spy.index[i].dayofweek != 0:  # 只看周一
        continue
    day_ret = float(spy['day_ret'].iloc[i])
    if day_ret >= -0.01:  # 周一没跌超1%
        continue
    
    entry_price = float(spy['Close'].iloc[i])  # 周一收盘买入
    # 周二数据
    next_idx = i + 1
    if next_idx >= len(spy):
        break
    exit_price = float(spy['Close'].iloc[next_idx])
    high_tue = float(spy['High'].iloc[next_idx])
    low_tue = float(spy['Low'].iloc[next_idx])
    
    pnl = (exit_price - entry_price) / entry_price * 100
    max_gain = (high_tue - entry_price) / entry_price * 100
    max_loss = (low_tue - entry_price) / entry_price * 100
    
    trades_tue.append({
        'pnl_pct': pnl,
        'max_gain_pct': max_gain,
        'max_loss_pct': max_loss,
    })

tdf = pd.DataFrame(trades_tue)
print(f"  {len(tdf)} 笔交易")
print(f"  胜率: {(tdf['pnl_pct']>0).mean()*100:.1f}%, 均收: {tdf['pnl_pct'].mean():+.3f}%")
print(f"\n  盘中最大浮盈: 中位 +{tdf['max_gain_pct'].median():.2f}%, 75th +{tdf['max_gain_pct'].quantile(0.75):.2f}%")
print(f"  盘中最大浮亏: 中位 {tdf['max_loss_pct'].median():.2f}%, 25th {tdf['max_loss_pct'].quantile(0.25):.2f}%")

print(f"\n  止盈分析:")
for tp in [1, 1.5, 2, 3]:
    would_trigger = (tdf['max_gain_pct'] >= tp).sum()
    pct_trigger = would_trigger / len(tdf) * 100
    triggered = tdf[tdf['max_gain_pct'] >= tp]
    not_triggered = tdf[tdf['max_gain_pct'] < tp]
    simulated = (len(triggered) * tp + not_triggered['pnl_pct'].sum()) / len(tdf) if len(tdf) > 0 else 0
    print(f"    止盈 {tp:.1f}%: {pct_trigger:.1f}% 触发, 均收 {tdf['pnl_pct'].mean():+.3f}% → {simulated:+.3f}%")

print(f"\n  止损分析:")
for sl in [1, 2, 3, 5]:
    would_stop = (tdf['max_loss_pct'] <= -sl).sum()
    pct_stop = would_stop / len(tdf) * 100
    stopped = tdf[tdf['max_loss_pct'] <= -sl]
    not_stopped = tdf[tdf['max_loss_pct'] > -sl]
    simulated = (len(stopped) * (-sl) + not_stopped['pnl_pct'].sum()) / len(tdf) if len(tdf) > 0 else 0
    print(f"    止损-{sl}%: {pct_stop:.1f}% 触发, 均收 {tdf['pnl_pct'].mean():+.3f}% → {simulated:+.3f}%")

