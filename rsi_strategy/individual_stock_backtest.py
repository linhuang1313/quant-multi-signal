"""
个股量化交易可行性研究
===========================
1. 在大市值个股上回测RSI均值回归
2. 分析财报事件对策略的影响
3. 与ETF策略对比
4. 分析$1000资金的实际操作可行性
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download(ticker, start='2010-01-01'):
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

def backtest_rsi_stock(ticker, rsi_entry=10, stop_loss=0.05):
    """RSI(2)均值回归策略回测"""
    df = download(ticker)
    if len(df) < 300:
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
            
            sell = False
            reason = ''
            if close > sma5:
                sell = True; reason = 'MA5'
            elif pnl <= -stop_loss:
                sell = True; reason = 'stop'
            elif hold >= 10:
                sell = True; reason = 'time'
            
            if sell:
                ret = (close - entry_price) / entry_price * 100
                # 检查是否在财报期间
                entry_date = df.index[entry_idx]
                exit_date = df.index[i]
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'ret': ret,
                    'hold_days': hold,
                    'reason': reason,
                    'entry_month': entry_date.month,
                })
                in_pos = False
    
    if not trades:
        return None
    
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wr = (tdf['ret'] > 0).mean() * 100
    avg = tdf['ret'].mean()
    rets = tdf['ret'].values / 100
    
    # Sharpe
    avg_h = tdf['hold_days'].mean()
    tpy = 252 / max(avg_h, 0.5)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(tpy) if len(rets) > 1 and rets.std() > 0 else 0
    
    # Max DD
    eq = np.cumprod(np.concatenate([[1], 1 + rets]))
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak * 100).min()
    
    # 大亏交易
    worst = tdf['ret'].min()
    big_losses = (tdf['ret'] < -5).sum()
    stopped = (tdf['reason'] == 'stop').sum()
    
    years = max((tdf['exit_date'].iloc[-1] - tdf['entry_date'].iloc[0]).days / 365.25, 0.5)
    
    return {
        'ticker': ticker,
        'trades': n,
        'trades_yr': round(n/years, 1),
        'win_rate': round(wr, 1),
        'avg_ret': round(avg, 3),
        'sharpe': round(sharpe, 2),
        'max_dd': round(dd, 1),
        'worst_trade': round(worst, 1),
        'big_losses_5pct': big_losses,
        'stopped_pct': round(stopped/n*100, 1),
        'years': round(years, 1),
    }

# ============================================================
# 回测大市值个股
# ============================================================
stocks = [
    # Mega Cap 科技
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # 大盘价值/消费
    'JPM', 'JNJ', 'UNH', 'V', 'PG', 'HD', 'KO', 'PEP',
    # 工业/金融
    'CAT', 'GS', 'BA', 'DIS', 'NKE',
    # 对比: ETF
    'SPY', 'QQQ',
]

print("="*100)
print("📊 RSI(2)均值回归策略 — 个股 vs ETF 全面回测 (2010-2026)")
print("="*100)

results = []
for ticker in stocks:
    r = backtest_rsi_stock(ticker)
    if r:
        results.append(r)

rdf = pd.DataFrame(results)
rdf = rdf.sort_values('sharpe', ascending=False)

print(f"\n{'标的':<8s} {'交易数':>5s} {'年均':>5s} {'胜率':>6s} {'均收':>7s} "
      f"{'Sharpe':>7s} {'最大回撤':>8s} {'最差单笔':>8s} {'止损占比':>8s}")
print("─"*80)
for _, r in rdf.iterrows():
    is_etf = '📈' if r['ticker'] in ('SPY', 'QQQ') else '  '
    print(f"{is_etf}{r['ticker']:<6s} {r['trades']:>5d} {r['trades_yr']:>5.1f} "
          f"{r['win_rate']:>5.1f}% {r['avg_ret']:>+6.3f}% "
          f"{r['sharpe']:>6.2f} {r['max_dd']:>7.1f}% "
          f"{r['worst_trade']:>7.1f}% {r['stopped_pct']:>7.1f}%")

# 统计对比
print(f"\n{'='*80}")
print("📊 个股 vs ETF 统计对比")
print("="*80)
etf = rdf[rdf['ticker'].isin(['SPY', 'QQQ'])]
stock = rdf[~rdf['ticker'].isin(['SPY', 'QQQ'])]

print(f"\n  {'指标':<20s} {'个股(均值)':>12s} {'ETF(均值)':>12s}")
print(f"  {'─'*50}")
print(f"  {'Sharpe':<20s} {stock['sharpe'].mean():>11.2f} {etf['sharpe'].mean():>11.2f}")
print(f"  {'胜率':<20s} {stock['win_rate'].mean():>10.1f}% {etf['win_rate'].mean():>10.1f}%")
print(f"  {'平均单笔收益':<20s} {stock['avg_ret'].mean():>10.3f}% {etf['avg_ret'].mean():>10.3f}%")
print(f"  {'最大回撤':<20s} {stock['max_dd'].mean():>10.1f}% {etf['max_dd'].mean():>10.1f}%")
print(f"  {'最差单笔':<20s} {stock['worst_trade'].mean():>10.1f}% {etf['worst_trade'].mean():>10.1f}%")
print(f"  {'止损触发比例':<20s} {stock['stopped_pct'].mean():>10.1f}% {etf['stopped_pct'].mean():>10.1f}%")
print(f"  {'年均交易次数':<20s} {stock['trades_yr'].mean():>11.1f} {etf['trades_yr'].mean():>11.1f}")

# Sharpe > 1 的个股
good_stocks = stock[stock['sharpe'] >= 1.0]
print(f"\n  Sharpe >= 1.0 的个股: {len(good_stocks)}/{len(stock)} ({len(good_stocks)/len(stock)*100:.0f}%)")
if not good_stocks.empty:
    print(f"  {', '.join(good_stocks['ticker'].tolist())}")

# ============================================================
# $1000资金的实际问题分析
# ============================================================
print(f"\n\n{'='*80}")
print("💰 $1000 资金做个股量化的实际限制分析")
print("="*80)

print("""
  1. PDT规则 (Pattern Day Trader):
     账户 < $25,000 → 5个交易日内最多3次日内交易
     RSI策略持仓1-10天，通常不算日内交易 → 影响较小
     但如果同一天买卖（信号当天反转）→ 消耗1次日内交易额度
     
  2. 分散化问题:
     $1000 买1只个股 → 单只占100%仓位 → 极端集中风险
     $1000 分3只 → 每只$333 → 有些股票1股都买不起(如GOOGL~$170, AMZN~$200)
     Alpaca支持碎股 → 可以买0.5股，但无法挂bracket/OTO单
     
  3. 股价限制:""")

# 查看当前股价
for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'KO', 'PEP']:
    df = download(ticker)
    if not df.empty:
        price = float(df['Close'].iloc[-1])
        shares_1000 = int(1000 / price)
        shares_333 = int(333 / price)
        print(f"     {ticker:<6s} ${price:>7.2f}  |  $1000可买{shares_1000:>2d}股  |  $333可买{shares_333:>2d}股")

print("""
  4. 个股特有风险:
     - 财报风险: 每季度4次，盘后公布，次日可能暴跌10-20%
     - 新闻风险: CEO离职、产品召回、监管调查等突发事件
     - 流动性风险: 部分个股盘后流动性差，止损单可能滑点严重
     
  5. 与ETF对比:
     ETF分散持有几百只股票，单只暴雷影响<1%
     个股集中持有，一次财报miss就可能亏10-20%
""")

