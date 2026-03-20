"""
RSI均值回归 + Turnaround Tuesday 组合回测
==========================================
模拟两个策略共享同一账户、同时运行的真实表现

关键点:
1. 共享资金池（初始$100,000）
2. 每个策略每只标的最多占40%权益
3. 资金不足时跳过信号
4. RSI和Tuesday可能同时持仓同一标的 → 用独立tracking避免冲突
5. 逐日模拟，精确计算资金占用和权益曲线
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


def download_data(ticker, start='1999-03-10'):
    """QQQ从1999-03-10开始，取两者共同区间"""
    df = yf.download(ticker, start=start, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def run_combined_backtest(initial_capital=100000, position_pct=0.40, commission=0.0003):
    """
    逐日模拟两策略共享账户
    """
    print("=" * 75)
    print("📊 RSI + Turnaround Tuesday 组合回测")
    print("=" * 75)
    
    # ── 下载数据 ──
    print("\n下载数据...")
    spy = download_data('SPY', '1999-03-10')
    qqq = download_data('QQQ', '1999-03-10')
    
    # 对齐日期
    common_dates = spy.index.intersection(qqq.index)
    spy = spy.loc[common_dates]
    qqq = qqq.loc[common_dates]
    
    print(f"  共同区间: {common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  交易日数: {len(common_dates)}")
    years = (common_dates[-1] - common_dates[0]).days / 365.25
    print(f"  约 {years:.1f} 年")
    
    # ── 预计算指标 ──
    for df in [spy, qqq]:
        df['RSI2'] = calc_rsi(df['Close'], 2)
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['SMA5'] = df['Close'].rolling(5).mean()
        df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['weekday'] = df.index.weekday
        df['prev_close'] = df['Close'].shift(1)
    
    # ── RSI策略参数 ──
    rsi_params = {
        'SPY': {'rsi_entry': 10, 'use_ibs': True, 'ibs_threshold': 0.3},
        'QQQ': {'rsi_entry': 15, 'use_ibs': False, 'ibs_threshold': 1.0},
    }
    
    # ── 模拟引擎 ──
    cash = initial_capital
    
    # 持仓追踪 (分策略)
    # rsi_positions: {ticker: {entry_price, qty, entry_idx, entry_date}}
    # tue_positions: {ticker: {entry_price, qty, entry_idx, entry_date}}
    rsi_positions = {}
    tue_positions = {}
    
    # 交易记录
    rsi_trades = []
    tue_trades = []
    
    # 权益曲线
    equity_curve = []
    dates_curve = []
    
    # 每日暴露追踪
    rsi_exposure_days = 0
    tue_exposure_days = 0
    
    data = {'SPY': spy, 'QQQ': qqq}
    
    for i in range(200, len(common_dates)):
        date = common_dates[i]
        
        # ── 计算当前权益 ──
        holdings_value = 0
        for ticker, pos in {**rsi_positions, **tue_positions}.items():
            # 注意: 如果RSI和Tuesday同时持有同一ticker, 这里会重复计算
            # 但在实际中我们避免了这种情况
            pass
        
        # 精确计算持仓市值
        rsi_value = 0
        for ticker, pos in rsi_positions.items():
            current_price = float(data[ticker]['Close'].iloc[i])
            rsi_value += pos['qty'] * current_price
        
        tue_value = 0
        for ticker, pos in tue_positions.items():
            current_price = float(data[ticker]['Close'].iloc[i])
            tue_value += pos['qty'] * current_price
        
        total_equity = cash + rsi_value + tue_value
        equity_curve.append(total_equity)
        dates_curve.append(date)
        
        if rsi_positions:
            rsi_exposure_days += 1
        if tue_positions:
            tue_exposure_days += 1
        
        # ══════════════════════════════════════
        # RSI策略: 出场检查
        # ══════════════════════════════════════
        for ticker in list(rsi_positions.keys()):
            pos = rsi_positions[ticker]
            df = data[ticker]
            current_price = float(df['Close'].iloc[i])
            sma5 = float(df['SMA5'].iloc[i])
            entry_price = pos['entry_price']
            hold_days = i - pos['entry_idx']
            pnl_pct = (current_price - entry_price) / entry_price
            
            sell = False
            reason = ""
            
            # 正常出场: 价格 > MA5
            if current_price > sma5:
                sell = True
                reason = "价格>MA5"
            # 硬止损: -5%
            elif pnl_pct <= -0.05:
                sell = True
                reason = "止损-5%"
            # 时间止损: 10天
            elif hold_days >= 10:
                sell = True
                reason = f"时间止损{hold_days}天"
            
            if sell:
                proceeds = pos['qty'] * current_price * (1 - commission)
                cash += proceeds
                net_ret = (current_price - entry_price) / entry_price - 2 * commission
                rsi_trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'qty': pos['qty'],
                    'return_pct': net_ret * 100,
                    'pnl_usd': pos['qty'] * (current_price - entry_price),
                    'hold_days': hold_days,
                    'reason': reason,
                })
                del rsi_positions[ticker]
        
        # ══════════════════════════════════════
        # Tuesday策略: 周二平仓
        # ══════════════════════════════════════
        weekday = date.weekday()
        
        if weekday == 1:  # Tuesday
            for ticker in list(tue_positions.keys()):
                pos = tue_positions[ticker]
                df = data[ticker]
                current_price = float(df['Close'].iloc[i])
                entry_price = pos['entry_price']
                hold_days = i - pos['entry_idx']
                
                proceeds = pos['qty'] * current_price * (1 - commission)
                cash += proceeds
                net_ret = (current_price - entry_price) / entry_price - 2 * commission
                tue_trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'qty': pos['qty'],
                    'return_pct': net_ret * 100,
                    'pnl_usd': pos['qty'] * (current_price - entry_price),
                    'hold_days': hold_days,
                })
                del tue_positions[ticker]
        
        # ══════════════════════════════════════
        # RSI策略: 入场检查
        # ══════════════════════════════════════
        for ticker in ['SPY', 'QQQ']:
            if ticker in rsi_positions:
                continue
            
            df = data[ticker]
            params = rsi_params[ticker]
            
            close = float(df['Close'].iloc[i])
            rsi2 = float(df['RSI2'].iloc[i])
            sma200 = float(df['SMA200'].iloc[i])
            ibs = float(df['IBS'].iloc[i])
            
            above_ma200 = close > sma200
            rsi_oversold = rsi2 < params['rsi_entry']
            ibs_ok = (not params['use_ibs']) or (ibs < params['ibs_threshold'])
            
            if above_ma200 and rsi_oversold and ibs_ok:
                target_value = total_equity * position_pct
                if cash >= target_value * 0.5:
                    buy_value = min(target_value, cash * 0.90)
                    qty = max(1, int(buy_value / close))
                    cost = qty * close * (1 + commission)
                    if cost <= cash:
                        cash -= cost
                        rsi_positions[ticker] = {
                            'entry_price': close,
                            'qty': qty,
                            'entry_idx': i,
                            'entry_date': date,
                        }
        
        # ══════════════════════════════════════
        # Tuesday策略: 周一买入
        # ══════════════════════════════════════
        if weekday == 0:  # Monday
            for ticker in ['SPY', 'QQQ']:
                if ticker in tue_positions:
                    continue
                # 如果RSI也持有同一ticker, 跳过
                if ticker in rsi_positions:
                    continue
                
                df = data[ticker]
                close = float(df['Close'].iloc[i])
                prev_close = float(df['prev_close'].iloc[i])
                
                if pd.isna(prev_close) or prev_close == 0:
                    continue
                
                drop_pct = (close - prev_close) / prev_close
                
                if drop_pct <= -0.01:  # 跌1%+
                    target_value = total_equity * position_pct
                    if cash >= target_value * 0.3:
                        buy_value = min(target_value, cash * 0.90)
                        qty = max(1, int(buy_value / close))
                        cost = qty * close * (1 + commission)
                        if cost <= cash:
                            cash -= cost
                            tue_positions[ticker] = {
                                'entry_price': close,
                                'qty': qty,
                                'entry_idx': i,
                                'entry_date': date,
                            }
    
    # ══════════════════════════════════════
    # 统计汇总
    # ══════════════════════════════════════
    
    equity = np.array(equity_curve)
    total_days = len(equity)
    
    # Buy & Hold 对比 (SPY)
    spy_start = float(spy['Close'].iloc[200])
    spy_end = float(spy['Close'].iloc[-1])
    bh_total_ret = (spy_end / spy_start - 1) * 100
    bh_cagr = ((spy_end / spy_start) ** (1/years) - 1) * 100
    
    # SPY权益曲线 (buy & hold)
    spy_prices = spy['Close'].iloc[200:200+total_days].values.astype(float)
    spy_equity = initial_capital * (spy_prices / spy_prices[0])
    spy_peak = np.maximum.accumulate(spy_equity)
    spy_dd = ((spy_equity - spy_peak) / spy_peak * 100).min()
    
    # 组合统计
    final_equity = equity[-1]
    total_ret = (final_equity / initial_capital - 1) * 100
    cagr = ((final_equity / initial_capital) ** (1/years) - 1) * 100
    
    # 最大回撤
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_dd = drawdown.min()
    
    # 日收益率 (基于权益曲线)
    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # 暴露度
    rsi_exposure = rsi_exposure_days / total_days * 100
    tue_exposure = tue_exposure_days / total_days * 100
    
    # RSI交易统计
    rsi_df = pd.DataFrame(rsi_trades) if rsi_trades else pd.DataFrame()
    tue_df = pd.DataFrame(tue_trades) if tue_trades else pd.DataFrame()
    
    print(f"\n{'='*75}")
    print(f"📊 组合回测结果 ({years:.1f}年)")
    print(f"{'='*75}")
    
    print(f"\n  初始资金: ${initial_capital:,.0f}")
    print(f"  最终权益: ${final_equity:,.0f}")
    print(f"  总回报:   {total_ret:+.1f}%")
    print(f"  年化CAGR: {cagr:.2f}%")
    print(f"  Sharpe:   {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.1f}%")
    
    print(f"\n{'─'*75}")
    print(f"📈 对比 SPY 买入持有:")
    print(f"  SPY CAGR:   {bh_cagr:.2f}%")
    print(f"  SPY 总回报: {bh_total_ret:+.1f}%")
    print(f"  SPY 最大回撤: {spy_dd:.1f}%")
    
    print(f"\n{'─'*75}")
    print(f"📋 RSI均值回归:")
    if len(rsi_df) > 0:
        rsi_wr = (rsi_df['return_pct'] > 0).mean() * 100
        rsi_avg = rsi_df['return_pct'].mean()
        rsi_total_pnl = rsi_df['pnl_usd'].sum()
        print(f"  交易次数: {len(rsi_df)}")
        print(f"  胜率:     {rsi_wr:.1f}%")
        print(f"  平均回报: {rsi_avg:+.3f}%/笔")
        print(f"  总盈亏:   ${rsi_total_pnl:+,.0f}")
        print(f"  暴露度:   {rsi_exposure:.1f}%")
        # 按ticker分组
        for t in ['SPY', 'QQQ']:
            sub = rsi_df[rsi_df['ticker'] == t]
            if len(sub) > 0:
                print(f"    {t}: {len(sub)}笔  胜率{(sub['return_pct']>0).mean()*100:.1f}%  "
                      f"均回报{sub['return_pct'].mean():+.3f}%  总PnL ${sub['pnl_usd'].sum():+,.0f}")
    else:
        print(f"  无交易")
    
    print(f"\n{'─'*75}")
    print(f"📅 Turnaround Tuesday:")
    if len(tue_df) > 0:
        tue_wr = (tue_df['return_pct'] > 0).mean() * 100
        tue_avg = tue_df['return_pct'].mean()
        tue_total_pnl = tue_df['pnl_usd'].sum()
        print(f"  交易次数: {len(tue_df)}")
        print(f"  胜率:     {tue_wr:.1f}%")
        print(f"  平均回报: {tue_avg:+.3f}%/笔")
        print(f"  总盈亏:   ${tue_total_pnl:+,.0f}")
        print(f"  暴露度:   {tue_exposure:.1f}%")
        for t in ['SPY', 'QQQ']:
            sub = tue_df[tue_df['ticker'] == t]
            if len(sub) > 0:
                print(f"    {t}: {len(sub)}笔  胜率{(sub['return_pct']>0).mean()*100:.1f}%  "
                      f"均回报{sub['return_pct'].mean():+.3f}%  总PnL ${sub['pnl_usd'].sum():+,.0f}")
    else:
        print(f"  无交易")
    
    print(f"\n{'─'*75}")
    print(f"⏱️ 总暴露度: {rsi_exposure + tue_exposure:.1f}% (RSI {rsi_exposure:.1f}% + Tue {tue_exposure:.1f}%)")
    total_trades = len(rsi_df) + len(tue_df)
    print(f"📊 总交易次数: {total_trades} (RSI {len(rsi_df)} + Tue {len(tue_df)})")
    print(f"📊 年均交易: {total_trades/years:.1f}次")
    
    # 年度明细
    print(f"\n{'─'*75}")
    print(f"📅 年度表现:")
    print(f"{'年份':>6s} {'组合收益':>8s} {'SPY收益':>8s} {'RSI笔数':>7s} {'Tue笔数':>7s} {'最大回撤':>8s}")
    print(f"{'─'*75}")
    
    dates_arr = np.array(dates_curve)
    
    for year in range(dates_arr[0].year, dates_arr[-1].year + 1):
        year_mask = np.array([d.year == year for d in dates_arr])
        if not year_mask.any():
            continue
        
        year_equity = equity[year_mask]
        if len(year_equity) < 2:
            continue
        
        year_ret = (year_equity[-1] / year_equity[0] - 1) * 100
        
        year_spy = spy_equity[:total_days][year_mask]
        spy_year_ret = (year_spy[-1] / year_spy[0] - 1) * 100 if len(year_spy) >= 2 else 0
        
        year_peak = np.maximum.accumulate(year_equity)
        year_dd = ((year_equity - year_peak) / year_peak * 100).min()
        
        rsi_yr = len(rsi_df[rsi_df['exit_date'].apply(lambda d: d.year) == year]) if len(rsi_df) > 0 else 0
        tue_yr = len(tue_df[tue_df['exit_date'].apply(lambda d: d.year) == year]) if len(tue_df) > 0 else 0
        
        print(f"  {year:>4d}  {year_ret:>+7.2f}%  {spy_year_ret:>+7.2f}%  {rsi_yr:>5d}    {tue_yr:>5d}    {year_dd:>+7.1f}%")
    
    print(f"{'─'*75}")
    
    # 保存权益曲线数据
    eq_df = pd.DataFrame({
        'date': dates_curve,
        'combined_equity': equity_curve,
        'spy_bh_equity': spy_equity[:total_days].tolist(),
    })
    eq_df.to_csv('/home/user/workspace/quant-trading/rsi_strategy/data/combined_equity_curve.csv', index=False)
    print(f"\n✅ 权益曲线已保存")
    
    return {
        'years': round(years, 1),
        'total_return': round(total_ret, 1),
        'cagr': round(cagr, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_dd, 1),
        'total_trades': total_trades,
        'rsi_trades': len(rsi_df),
        'tue_trades': len(tue_df),
        'rsi_exposure': round(rsi_exposure, 1),
        'tue_exposure': round(tue_exposure, 1),
        'spy_cagr': round(bh_cagr, 2),
        'spy_max_dd': round(spy_dd, 1),
    }


if __name__ == '__main__':
    result = run_combined_backtest()
    print(f"\n\n汇总: {result}")
