"""
新策略回测: 从全网搜索中筛选的候选策略
==========================================
1. Lower Lows策略: 连续N天新低 → 买入 (均值回归变体)
2. 隔夜效应: 收盘买入 → 次日开盘卖出
3. Down Week策略: 周跌 → 买入持有一周
4. Pre-Holiday效应: 假日前1天买入 → 假日后1天卖出
5. SPY-TLT轮动: 月底看谁弱买谁(基金经理再平衡效应)
6. Triple RSI: 三重RSI条件过滤
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def download(ticker, start='1999-01-01'):
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


def stats(trades, label):
    if not trades:
        return {'strategy': label, 'trades': 0}
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wr = (tdf['ret'] > 0).mean() * 100
    avg = tdf['ret'].mean()
    rets = tdf['ret'].values / 100
    cum = (1 + rets).prod()
    years = max((tdf['exit_date'].iloc[-1] - tdf['entry_date'].iloc[0]).days / 365.25, 0.5)
    cagr = (cum ** (1/years) - 1) * 100
    total_ret = (cum - 1) * 100
    
    avg_h = tdf['hold_days'].mean() if 'hold_days' in tdf.columns else 1
    tpy = 252 / max(avg_h, 0.5)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(tpy) if len(rets) > 1 and rets.std() > 0 else 0
    
    eq = np.cumprod(np.concatenate([[1], 1 + rets]))
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak * 100).min()
    
    total_hold = tdf['hold_days'].sum() if 'hold_days' in tdf.columns else n
    total_cal = (tdf['exit_date'].iloc[-1] - tdf['entry_date'].iloc[0]).days
    exposure = total_hold / (total_cal * 5/7) * 100 if total_cal > 0 else 0
    
    gp = tdf[tdf['ret'] > 0]['ret'].sum()
    gl = abs(tdf[tdf['ret'] < 0]['ret'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    
    return {
        'strategy': label, 'trades': n, 'years': round(years, 1),
        'win_rate': round(wr, 1), 'avg_ret': round(avg, 3),
        'cagr': round(cagr, 2), 'total_ret': round(total_ret, 1),
        'sharpe': round(sharpe, 2), 'max_dd': round(dd, 1),
        'exposure': round(exposure, 1), 'pf': round(pf, 2),
        'trades_yr': round(n/years, 1),
    }


def print_result(r):
    if r['trades'] == 0:
        print(f"  {r['strategy']:<40s}  无交易")
        return
    print(f"  {r['strategy']:<40s}  {r['trades']:>4d}笔  "
          f"胜率{r['win_rate']:>5.1f}%  均{r['avg_ret']:>+6.3f}%  "
          f"Sharpe {r['sharpe']:>5.2f}  回撤{r['max_dd']:>6.1f}%  "
          f"暴露{r['exposure']:>5.1f}%  CAGR{r['cagr']:>6.2f}%")


# ============================================================
# 1. Lower Lows + Lower Highs 策略
# 连续N天High<前一天High → 买入 → 收盘>前日High卖出
# ============================================================
def backtest_lower_highs(ticker, n_days=3, commission=0.0003):
    df = download(ticker)
    if len(df) < 300:
        return stats([], f"LowerHighs({n_days}) {ticker}")
    
    trades = []
    in_pos = False
    entry_price = entry_idx = 0
    entry_date = None
    
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    for i in range(201, len(df)):
        close = float(df['Close'].iloc[i])
        sma200 = float(df['SMA200'].iloc[i])
        
        if not in_pos:
            # 检查连续N天lower highs
            if close <= sma200:
                continue
            ok = True
            for j in range(n_days):
                if i - j - 1 < 0:
                    ok = False
                    break
                if float(df['High'].iloc[i-j]) >= float(df['High'].iloc[i-j-1]):
                    ok = False
                    break
            if ok:
                entry_price = close
                entry_date = df.index[i]
                entry_idx = i
                in_pos = True
        else:
            hold = i - entry_idx
            prev_high = float(df['High'].iloc[i-1])
            pnl = (close - entry_price) / entry_price
            
            sell = False
            if close > prev_high:  # 收盘>前日高点
                sell = True
            elif pnl <= -0.05:
                sell = True
            elif hold >= 10:
                sell = True
            
            if sell:
                ret = (close - entry_price) / entry_price * 100 - 2 * commission * 100
                trades.append({'entry_date': entry_date, 'exit_date': df.index[i], 'ret': ret, 'hold_days': hold})
                in_pos = False
    
    return stats(trades, f"LowerHighs({n_days}) {ticker}")


# ============================================================
# 2. 隔夜效应: 收盘买入 → 次日开盘卖出
# 条件: IBS < 0.5 + 价格 < 10日高点附近
# ============================================================
def backtest_overnight(ticker, commission=0.0003):
    df = download(ticker)
    if len(df) < 300:
        return stats([], f"Overnight {ticker}")
    
    df['ATR'] = (df['High'] - df['Low']).rolling(25).mean()
    df['High10'] = df['High'].rolling(10).max()
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    trades = []
    
    for i in range(26, len(df) - 1):
        close = float(df['Close'].iloc[i])
        ibs = float(df['IBS'].iloc[i])
        atr = float(df['ATR'].iloc[i])
        high10 = float(df['High10'].iloc[i])
        
        # 条件: 收盘价 < 10日高点 - ATR, IBS < 0.5
        band = high10 - atr
        if close < band and ibs < 0.5:
            next_open = float(df['Open'].iloc[i+1])
            ret = (next_open - close) / close * 100 - 2 * commission * 100
            trades.append({
                'entry_date': df.index[i],
                'exit_date': df.index[i+1],
                'ret': ret,
                'hold_days': 1,
            })
    
    return stats(trades, f"Overnight {ticker}")


# ============================================================
# 3. Down Week 策略: 周跌 → 持有一周
# ============================================================
def backtest_down_week(ticker, commission=0.0003):
    df = download(ticker)
    if len(df) < 500:
        return stats([], f"DownWeek {ticker}")
    
    # 按周聚合
    weekly = df['Close'].resample('W-FRI').last().dropna()
    
    trades = []
    for i in range(1, len(weekly) - 1):
        this_close = float(weekly.iloc[i])
        prev_close = float(weekly.iloc[i-1])
        
        if this_close < prev_close:  # 本周跌了
            next_close = float(weekly.iloc[i+1])
            ret = (next_close - this_close) / this_close * 100 - 2 * commission * 100
            trades.append({
                'entry_date': weekly.index[i],
                'exit_date': weekly.index[i+1],
                'ret': ret,
                'hold_days': 5,
            })
    
    return stats(trades, f"DownWeek {ticker}")


# ============================================================
# 4. Triple RSI: RSI(2)<5 + RSI(3)<20 + RSI(4)<30
# ============================================================
def backtest_triple_rsi(ticker, commission=0.0003):
    df = download(ticker)
    if len(df) < 300:
        return stats([], f"TripleRSI {ticker}")
    
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['RSI3'] = calc_rsi(df['Close'], 3)
    df['RSI4'] = calc_rsi(df['Close'], 4)
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA5'] = df['Close'].rolling(5).mean()
    
    trades = []
    in_pos = False
    entry_price = entry_idx = 0
    entry_date = None
    
    for i in range(201, len(df)):
        close = float(df['Close'].iloc[i])
        sma200 = float(df['SMA200'].iloc[i])
        sma5 = float(df['SMA5'].iloc[i])
        rsi2 = float(df['RSI2'].iloc[i])
        rsi3 = float(df['RSI3'].iloc[i])
        rsi4 = float(df['RSI4'].iloc[i])
        
        if not in_pos:
            if close > sma200 and rsi2 < 5 and rsi3 < 20 and rsi4 < 30:
                entry_price = close
                entry_date = df.index[i]
                entry_idx = i
                in_pos = True
        else:
            hold = i - entry_idx
            pnl = (close - entry_price) / entry_price
            sell = False
            if close > sma5: sell = True
            elif pnl <= -0.05: sell = True
            elif hold >= 10: sell = True
            
            if sell:
                ret = (close - entry_price) / entry_price * 100 - 2 * commission * 100
                trades.append({'entry_date': entry_date, 'exit_date': df.index[i], 'ret': ret, 'hold_days': hold})
                in_pos = False
    
    return stats(trades, f"TripleRSI {ticker}")


# ============================================================
# 5. SPY-TLT 月末轮动 (再平衡效应)
# 月中看SPY和TLT谁跑得差，月末买它，下月初卖
# ============================================================
def backtest_rebalance_effect(commission=0.0003):
    spy = download('SPY')
    tlt = download('TLT', start='2002-07-30')
    common = spy.index.intersection(tlt.index)
    spy = spy.loc[common]
    tlt = tlt.loc[common]
    
    # 月度数据
    spy_m = spy['Close'].resample('M').last()
    tlt_m = tlt['Close'].resample('M').last()
    spy_mid = spy['Close'].resample('M').apply(lambda x: x.iloc[len(x)//2] if len(x) > 0 else np.nan)
    tlt_mid = tlt['Close'].resample('M').apply(lambda x: x.iloc[len(x)//2] if len(x) > 0 else np.nan)
    
    # 对齐
    combined = pd.DataFrame({'spy_m': spy_m, 'tlt_m': tlt_m, 'spy_mid': spy_mid, 'tlt_mid': tlt_mid}).dropna()
    
    trades = []
    for i in range(1, len(combined) - 1):
        # 月中到月末的表现
        spy_half_ret = (float(combined['spy_m'].iloc[i]) - float(combined['spy_mid'].iloc[i])) / float(combined['spy_mid'].iloc[i])
        tlt_half_ret = (float(combined['tlt_m'].iloc[i]) - float(combined['tlt_mid'].iloc[i])) / float(combined['tlt_mid'].iloc[i])
        
        # 买表现差的(基金经理会再平衡买入)
        if spy_half_ret < tlt_half_ret:
            ticker_buy = 'SPY'
            entry_price = float(combined['spy_m'].iloc[i])
            # 简化: 用下月初几天的收益
            exit_price = float(combined['spy_mid'].iloc[i+1])
        else:
            ticker_buy = 'TLT'
            entry_price = float(combined['tlt_m'].iloc[i])
            exit_price = float(combined['tlt_mid'].iloc[i+1])
        
        ret = (exit_price - entry_price) / entry_price * 100 - 2 * commission * 100
        trades.append({
            'entry_date': combined.index[i],
            'exit_date': combined.index[i+1],
            'ret': ret,
            'hold_days': 10,
            'ticker': ticker_buy,
        })
    
    return stats(trades, "SPY-TLT Rebalance")


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    all_results = []
    tickers = ['SPY', 'QQQ', 'DIA', 'IWM']
    
    # 1. Lower Highs
    print("=" * 90)
    print("📊 1. Lower Highs 连续新低策略")
    print("=" * 90)
    for t in tickers:
        for n in [3, 4]:
            r = backtest_lower_highs(t, n)
            print_result(r)
            if r['trades'] >= 30: all_results.append(r)
    
    # 2. 隔夜效应
    print(f"\n{'='*90}")
    print("📊 2. 隔夜效应 (收盘买→次日开盘卖)")
    print("=" * 90)
    for t in ['SPY', 'QQQ', 'DIA', 'IWM', 'TLT', 'GLD']:
        r = backtest_overnight(t)
        print_result(r)
        if r['trades'] >= 30: all_results.append(r)
    
    # 3. Down Week
    print(f"\n{'='*90}")
    print("📊 3. Down Week 周跌买入策略")
    print("=" * 90)
    for t in tickers:
        r = backtest_down_week(t)
        print_result(r)
        if r['trades'] >= 30: all_results.append(r)
    
    # 4. Triple RSI
    print(f"\n{'='*90}")
    print("📊 4. Triple RSI 三重过滤")
    print("=" * 90)
    for t in ['SPY', 'QQQ', 'DIA', 'IWM', 'GLD']:
        r = backtest_triple_rsi(t)
        print_result(r)
        if r['trades'] >= 20: all_results.append(r)  # Triple RSI交易少，门槛放低
    
    # 5. SPY-TLT 再平衡
    print(f"\n{'='*90}")
    print("📊 5. SPY-TLT 月末再平衡效应")
    print("=" * 90)
    r = backtest_rebalance_effect()
    print_result(r)
    if r['trades'] >= 30: all_results.append(r)
    
    # 综合排名
    print(f"\n\n{'='*90}")
    print("🏆 综合排名 (按Sharpe)")
    print("=" * 90)
    valid = [r for r in all_results if r.get('sharpe', 0) > 0]
    valid.sort(key=lambda x: x['sharpe'], reverse=True)
    
    print(f"\n{'排名':>4s} {'策略':<40s} {'笔数':>5s} {'胜率':>6s} {'均回报':>7s} "
          f"{'Sharpe':>7s} {'回撤':>6s} {'暴露':>6s} {'CAGR':>7s}")
    print("─" * 100)
    for i, r in enumerate(valid[:20], 1):
        print(f"  {i:>2d}  {r['strategy']:<40s} {r['trades']:>5d} {r['win_rate']:>5.1f}% "
              f"{r['avg_ret']:>+6.3f}% {r['sharpe']:>6.2f} {r['max_dd']:>5.1f}% "
              f"{r['exposure']:>5.1f}% {r['cagr']:>6.2f}%")
    print("─" * 100)
