"""
跨资产多策略回测引擎
========================
在所有Alpaca可交易ETF上测试多种策略:
1. RSI(2) 均值回归 (已验证在SPY/QQQ上有效)
2. 月末效应 (Turn of Month)
3. Dual Momentum 动量轮动
4. 波动率均值回归 (VIX相关)
5. TLT月度季节性
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 数据下载
# ============================================================
def download(ticker, start='1999-01-01'):
    df = yf.download(ticker, start=start, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['Close'])
    return df


def calc_rsi(series, period=2):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_stats(trades_df, label=""):
    """从交易DataFrame计算统计指标"""
    if trades_df.empty or len(trades_df) == 0:
        return {'strategy': label, 'trades': 0}
    
    tdf = trades_df
    n = len(tdf)
    wr = (tdf['ret'] > 0).mean() * 100
    avg_ret = tdf['ret'].mean()
    
    cum = (1 + tdf['ret']/100).prod()
    years = max((tdf['exit_date'].iloc[-1] - tdf['entry_date'].iloc[0]).days / 365.25, 0.5)
    cagr = (cum ** (1/years) - 1) * 100
    total_ret = (cum - 1) * 100
    
    # Sharpe
    rets = tdf['ret'].values / 100
    avg_h = tdf['hold_days'].mean() if 'hold_days' in tdf.columns else 1
    tpy = 252 / max(avg_h, 0.5)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(tpy) if len(rets) > 1 and rets.std() > 0 else 0
    
    # Drawdown
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak * 100).min()
    
    # Exposure
    total_hold = tdf['hold_days'].sum() if 'hold_days' in tdf.columns else n
    total_cal = (tdf['exit_date'].iloc[-1] - tdf['entry_date'].iloc[0]).days
    exposure = total_hold / (total_cal * 5/7) * 100 if total_cal > 0 else 0
    
    # Profit Factor
    gp = tdf[tdf['ret'] > 0]['ret'].sum()
    gl = abs(tdf[tdf['ret'] < 0]['ret'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    
    return {
        'strategy': label,
        'trades': n,
        'years': round(years, 1),
        'win_rate': round(wr, 1),
        'avg_ret': round(avg_ret, 3),
        'cagr': round(cagr, 2),
        'total_ret': round(total_ret, 1),
        'sharpe': round(sharpe, 2),
        'max_dd': round(dd, 1),
        'exposure': round(exposure, 1),
        'profit_factor': round(pf, 2),
        'trades_per_year': round(n / years, 1),
    }


# ============================================================
# 策略1: RSI(2) 均值回归 — 在多资产上测试
# ============================================================
def backtest_rsi_multi(ticker, rsi_threshold=10, use_ibs=False, ibs_threshold=0.3, commission=0.0003):
    """在任意ETF上测试RSI(2)均值回归"""
    df = download(ticker)
    if len(df) < 250:
        return None
    
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    trades = []
    in_pos = False
    entry_price = 0
    entry_date = None
    entry_idx = 0
    
    for i in range(201, len(df)):
        close = float(df['Close'].iloc[i])
        rsi = float(df['RSI2'].iloc[i])
        sma200 = float(df['SMA200'].iloc[i])
        sma5 = float(df['SMA5'].iloc[i])
        ibs = float(df['IBS'].iloc[i])
        
        if not in_pos:
            above_ma = close > sma200
            rsi_low = rsi < rsi_threshold
            ibs_ok = (not use_ibs) or (ibs < ibs_threshold)
            
            if above_ma and rsi_low and ibs_ok:
                entry_price = close
                entry_date = df.index[i]
                entry_idx = i
                in_pos = True
        else:
            hold_days = i - entry_idx
            pnl_pct = (close - entry_price) / entry_price
            
            sell = False
            if close > sma5:
                sell = True
            elif pnl_pct <= -0.05:
                sell = True
            elif hold_days >= 10:
                sell = True
            
            if sell:
                ret = (close - entry_price) / entry_price * 100 - 2 * commission * 100
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'ret': ret,
                    'hold_days': hold_days,
                })
                in_pos = False
    
    if not trades:
        return None
    return compute_stats(pd.DataFrame(trades), f"RSI(2)<{rsi_threshold} {ticker}")


# ============================================================
# 策略2: 月末效应 (Turn of Month)
# ============================================================
def backtest_tom(ticker, entry_offset=-4, exit_offset=3, commission=0.0003):
    """
    月末效应: 在月末倒数第N天买入，月初第M天卖出
    entry_offset=-4: 月末倒数第4个交易日买入
    exit_offset=3: 月初第3个交易日卖出
    """
    df = download(ticker)
    if len(df) < 500:
        return None
    
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # 标记每月的交易日序号(从月末倒数)和(从月初正数)
    trades = []
    
    # 按月分组
    groups = df.groupby([df.index.year, df.index.month])
    
    months = sorted(groups.groups.keys())
    
    for idx in range(len(months) - 1):
        ym = months[idx]
        ym_next = months[idx + 1]
        
        this_month = groups.get_group(ym)
        next_month = groups.get_group(ym_next)
        
        # 月末倒数第N天
        if len(this_month) < abs(entry_offset):
            continue
        entry_row = this_month.iloc[entry_offset]
        entry_date = this_month.index[entry_offset]
        entry_price = float(entry_row['Close'])
        
        # 下月初第M天
        if len(next_month) < exit_offset:
            continue
        exit_row = next_month.iloc[exit_offset - 1]  # 0-indexed
        exit_date = next_month.index[exit_offset - 1]
        exit_price = float(exit_row['Close'])
        
        ret = (exit_price - entry_price) / entry_price * 100 - 2 * commission * 100
        hold_days = (exit_date - entry_date).days
        
        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ret': ret,
            'hold_days': max(hold_days, 1),
        })
    
    if not trades:
        return None
    return compute_stats(pd.DataFrame(trades), f"TOM({entry_offset},{exit_offset}) {ticker}")


# ============================================================
# 策略3: Dual Momentum 动量轮动
# ============================================================
def backtest_dual_momentum(assets=['SPY', 'EFA', 'BND'], lookback=12, commission=0.0003):
    """
    Dual Momentum:
    每月末检查，持有过去N月动量最强的资产
    如果最强资产动量<0，则持有债券(BND)
    """
    dfs = {}
    for ticker in assets:
        df = download(ticker)
        if len(df) < 300:
            return None
        dfs[ticker] = df['Close'].resample('M').last()
    
    # 对齐
    combined = pd.DataFrame(dfs).dropna()
    if len(combined) < lookback + 12:
        return None
    
    trades = []
    current_holding = None
    entry_price = 0
    entry_date = None
    
    for i in range(lookback, len(combined) - 1):
        date = combined.index[i]
        
        # 计算各资产过去N月动量
        momentums = {}
        for ticker in assets:
            current = float(combined[ticker].iloc[i])
            past = float(combined[ticker].iloc[i - lookback])
            momentums[ticker] = (current - past) / past * 100
        
        # 选最强
        best = max(assets[:-1], key=lambda t: momentums[t])  # 排除BND
        
        # 绝对动量检查
        if momentums[best] <= 0:
            target = assets[-1]  # 转入债券
        else:
            target = best
        
        # 换仓
        if target != current_holding:
            # 平旧仓
            if current_holding is not None:
                exit_price = float(combined[current_holding].iloc[i])
                ret = (exit_price - entry_price) / entry_price * 100 - 2 * commission * 100
                hold_months = i - trades[-1].get('entry_idx', i) if trades else 1
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'ret': ret,
                    'hold_days': max(hold_months * 21, 1),
                    'ticker': current_holding,
                })
            
            # 开新仓
            current_holding = target
            entry_price = float(combined[target].iloc[i])
            entry_date = date
    
    # 平最后一笔
    if current_holding:
        exit_price = float(combined[current_holding].iloc[-1])
        ret = (exit_price - entry_price) / entry_price * 100 - 2 * commission * 100
        trades.append({
            'entry_date': entry_date,
            'exit_date': combined.index[-1],
            'ret': ret,
            'hold_days': max((combined.index[-1] - entry_date).days, 1),
            'ticker': current_holding,
        })
    
    if not trades:
        return None
    return compute_stats(pd.DataFrame(trades), f"DualMom({lookback}m) {'/'.join(assets)}")


# ============================================================  
# 策略4: TLT月度季节性 (月末买入+月初做空)
# ============================================================
def backtest_tlt_seasonal(commission=0.0003):
    """
    TLT季节性策略 (来自学术研究):
    - 多头: 月末倒数第4天买入 → 月末平仓
    - 空头: 月初做空 → 月初第7天平仓
    这里只测多头部分(更安全)
    """
    return backtest_tom('TLT', entry_offset=-4, exit_offset=0, commission=commission)


# ============================================================
# 策略5: 波动率均值回归 (做空VIX spike)
# ============================================================
def backtest_vix_reversion(ticker='UVXY', rsi_threshold=85, commission=0.001):
    """
    波动率均值回归:
    VIX飙升后(RSI>85)做空UVXY → RSI回到50以下平仓
    注意: UVXY有时间衰减，长期做空有天然优势，但短期风险巨大
    这个策略风险极高，仅作参考
    """
    df = download(ticker, start='2011-10-01')
    if len(df) < 200:
        return None
    
    df['RSI14'] = calc_rsi(df['Close'], 14)
    
    trades = []
    in_pos = False
    entry_price = 0
    entry_date = None
    entry_idx = 0
    
    for i in range(50, len(df)):
        close = float(df['Close'].iloc[i])
        rsi14 = float(df['RSI14'].iloc[i])
        
        if not in_pos:
            if rsi14 > rsi_threshold:
                entry_price = close
                entry_date = df.index[i]
                entry_idx = i
                in_pos = True
        else:
            hold_days = i - entry_idx
            # 做空，所以收益是反向的
            pnl_pct = (entry_price - close) / entry_price
            
            sell = False
            if rsi14 < 50:
                sell = True
            elif pnl_pct <= -0.15:  # 止损15%
                sell = True
            elif hold_days >= 20:
                sell = True
            
            if sell:
                ret = (entry_price - close) / entry_price * 100 - 2 * commission * 100
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'ret': ret,
                    'hold_days': hold_days,
                })
                in_pos = False
    
    if not trades:
        return None
    return compute_stats(pd.DataFrame(trades), f"VIX_Short RSI>{rsi_threshold} {ticker}")


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    all_results = []
    
    # ── 1. RSI均值回归 在多资产上 ──
    print("=" * 85)
    print("📊 策略1: RSI(2) 均值回归 — 跨资产测试")
    print("=" * 85)
    
    rsi_tests = [
        # (ticker, rsi_threshold, use_ibs, ibs_threshold)
        ('SPY', 10, True, 0.3),
        ('QQQ', 15, False, 1.0),
        ('TLT', 15, False, 1.0),
        ('TLT', 10, False, 1.0),
        ('IEF', 15, False, 1.0),
        ('GLD', 15, False, 1.0),
        ('GLD', 10, False, 1.0),
        ('SLV', 15, False, 1.0),
        ('HYG', 15, False, 1.0),
        ('LQD', 15, False, 1.0),
        ('EFA', 15, False, 1.0),   # 国际股票
        ('EEM', 15, False, 1.0),   # 新兴市场
        ('IWM', 15, False, 1.0),   # 小盘股
        ('DIA', 10, True, 0.3),    # 道琼斯
        ('XLF', 15, False, 1.0),   # 金融
        ('XLE', 15, False, 1.0),   # 能源
        ('XLK', 15, False, 1.0),   # 科技
        ('XLV', 15, False, 1.0),   # 医疗
        ('TIP', 15, False, 1.0),   # 通胀债
    ]
    
    for ticker, rsi_th, use_ibs, ibs_th in rsi_tests:
        r = backtest_rsi_multi(ticker, rsi_th, use_ibs, ibs_th)
        if r and r['trades'] >= 30:
            all_results.append(r)
            print(f"  {r['strategy']:<25s}  {r['trades']:>4d}笔  "
                  f"胜率{r['win_rate']:>5.1f}%  均{r['avg_ret']:>+6.3f}%  "
                  f"Sharpe {r['sharpe']:>5.2f}  回撤{r['max_dd']:>6.1f}%  "
                  f"暴露{r['exposure']:>5.1f}%  CAGR{r['cagr']:>6.2f}%")
    
    # ── 2. 月末效应 ──
    print(f"\n{'='*85}")
    print("📊 策略2: 月末效应 (Turn of Month)")
    print("=" * 85)
    
    tom_tickers = ['SPY', 'QQQ', 'TLT', 'IEF', 'GLD', 'EFA', 'IWM', 'DIA']
    tom_params = [(-4, 3), (-3, 3), (-4, 2), (-2, 2)]
    
    for ticker in tom_tickers:
        for entry_off, exit_off in tom_params:
            r = backtest_tom(ticker, entry_off, exit_off)
            if r and r['trades'] >= 50:
                all_results.append(r)
                print(f"  {r['strategy']:<25s}  {r['trades']:>4d}笔  "
                      f"胜率{r['win_rate']:>5.1f}%  均{r['avg_ret']:>+6.3f}%  "
                      f"Sharpe {r['sharpe']:>5.2f}  回撤{r['max_dd']:>6.1f}%  "
                      f"暴露{r['exposure']:>5.1f}%  CAGR{r['cagr']:>6.2f}%")
    
    # ── 3. Dual Momentum ──
    print(f"\n{'='*85}")
    print("📊 策略3: Dual Momentum 动量轮动")
    print("=" * 85)
    
    dm_tests = [
        (['SPY', 'EFA', 'BND'], 12),
        (['SPY', 'EFA', 'TLT'], 12),
        (['QQQ', 'EFA', 'TLT'], 12),
        (['SPY', 'EFA', 'BND'], 6),
        (['SPY', 'GLD', 'TLT'], 12),
        (['SPY', 'EEM', 'TLT'], 12),
    ]
    
    for assets, lookback in dm_tests:
        r = backtest_dual_momentum(assets, lookback)
        if r:
            all_results.append(r)
            print(f"  {r['strategy']:<35s}  {r['trades']:>4d}笔  "
                  f"胜率{r['win_rate']:>5.1f}%  均{r['avg_ret']:>+6.3f}%  "
                  f"Sharpe {r['sharpe']:>5.2f}  回撤{r['max_dd']:>6.1f}%  "
                  f"CAGR{r['cagr']:>6.2f}%")
    
    # ── 4. VIX波动率策略 ──
    print(f"\n{'='*85}")
    print("📊 策略4: 波动率均值回归")
    print("=" * 85)
    
    for rsi_th in [80, 85, 90]:
        r = backtest_vix_reversion('UVXY', rsi_th)
        if r and r['trades'] >= 10:
            all_results.append(r)
            print(f"  {r['strategy']:<25s}  {r['trades']:>4d}笔  "
                  f"胜率{r['win_rate']:>5.1f}%  均{r['avg_ret']:>+6.3f}%  "
                  f"Sharpe {r['sharpe']:>5.2f}  回撤{r['max_dd']:>6.1f}%  "
                  f"暴露{r['exposure']:>5.1f}%")
    
    # ══════════════════════════════════════
    # 综合排名
    # ══════════════════════════════════════
    print(f"\n\n{'='*85}")
    print("🏆 综合排名 (按Sharpe排序，最低30笔交易)")
    print("=" * 85)
    
    # 过滤有效策略
    valid = [r for r in all_results if r.get('trades', 0) >= 30 and r.get('sharpe', 0) > 0]
    valid.sort(key=lambda x: x.get('sharpe', 0), reverse=True)
    
    print(f"\n{'排名':>4s} {'策略':<35s} {'交易':>5s} {'胜率':>6s} {'均回报':>7s} "
          f"{'Sharpe':>7s} {'回撤':>6s} {'暴露':>6s} {'CAGR':>7s}")
    print("─" * 95)
    
    for i, r in enumerate(valid[:30], 1):
        print(f"  {i:>2d}  {r['strategy']:<35s} {r['trades']:>5d} {r['win_rate']:>5.1f}% "
              f"{r['avg_ret']:>+6.3f}% {r['sharpe']:>6.2f} {r['max_dd']:>5.1f}% "
              f"{r.get('exposure', 0):>5.1f}% {r['cagr']:>6.2f}%")
    
    print("─" * 95)
    
    # 保存结果
    pd.DataFrame(valid).to_csv('/home/user/workspace/quant-trading/rsi_strategy/data/multi_asset_results.csv', index=False)
    print(f"\n✅ 结果已保存")
