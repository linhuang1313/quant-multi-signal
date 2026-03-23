"""
5策略组合回测: RSI(SPY+QQQ+GLD) + Tuesday(SPY+QQQ)
共享$100,000账户，逐日模拟
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


def download(ticker, start='2004-11-18'):
    """GLD从2004-11-18开始，取共同区间"""
    df = yf.download(ticker, start=start, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=['Close'])


def run():
    initial_capital = 100000
    commission = 0.0003

    print("下载数据...")
    spy = download('SPY')
    qqq = download('QQQ')
    gld = download('GLD')

    # 对齐
    common = spy.index.intersection(qqq.index).intersection(gld.index)
    spy = spy.loc[common]
    qqq = qqq.loc[common]
    gld = gld.loc[common]

    years = (common[-1] - common[0]).days / 365.25
    print(f"共同区间: {common[0].strftime('%Y-%m-%d')} ~ {common[-1].strftime('%Y-%m-%d')} ({years:.1f}年)")

    # 预计算指标
    data = {'SPY': spy, 'QQQ': qqq, 'GLD': gld}
    for df in data.values():
        df['RSI2'] = calc_rsi(df['Close'], 2)
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['SMA5'] = df['Close'].rolling(5).mean()
        df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['weekday'] = df.index.weekday
        df['prev_close'] = df['Close'].shift(1)

    # RSI参数
    rsi_params = {
        'SPY': {'rsi_entry': 10, 'use_ibs': True, 'ibs_threshold': 0.3, 'pos_pct': 0.40},
        'QQQ': {'rsi_entry': 15, 'use_ibs': False, 'ibs_threshold': 1.0, 'pos_pct': 0.40},
        'GLD': {'rsi_entry': 10, 'use_ibs': False, 'ibs_threshold': 1.0, 'pos_pct': 0.20},
    }

    # Tuesday参数
    tue_tickers = ['SPY', 'QQQ']
    tue_drop = 0.01
    tue_pos_pct = 0.40

    cash = initial_capital
    rsi_positions = {}
    tue_positions = {}
    rsi_trades = []
    tue_trades = []
    equity_curve = []
    dates_curve = []

    for i in range(200, len(common)):
        date = common[i]

        # 计算权益
        rsi_val = sum(pos['qty'] * float(data[t]['Close'].iloc[i]) for t, pos in rsi_positions.items())
        tue_val = sum(pos['qty'] * float(data[t]['Close'].iloc[i]) for t, pos in tue_positions.items())
        total_equity = cash + rsi_val + tue_val
        equity_curve.append(total_equity)
        dates_curve.append(date)

        weekday = date.weekday()

        # ── RSI出场 ──
        for ticker in list(rsi_positions.keys()):
            pos = rsi_positions[ticker]
            df = data[ticker]
            close = float(df['Close'].iloc[i])
            sma5 = float(df['SMA5'].iloc[i])
            entry_price = pos['entry_price']
            hold_days = i - pos['entry_idx']
            pnl_pct = (close - entry_price) / entry_price

            sell = False
            if close > sma5: sell = True
            elif pnl_pct <= -0.05: sell = True
            elif hold_days >= 10: sell = True

            if sell:
                cash += pos['qty'] * close * (1 - commission)
                ret = (close - entry_price) / entry_price * 100 - 2 * commission * 100
                rsi_trades.append({'ticker': ticker, 'entry_date': pos['entry_date'], 'exit_date': date, 'ret': ret, 'hold_days': hold_days, 'pnl_usd': pos['qty'] * (close - entry_price)})
                del rsi_positions[ticker]

        # ── Tuesday出场(周二) ──
        if weekday == 1:
            for ticker in list(tue_positions.keys()):
                pos = tue_positions[ticker]
                df = data[ticker]
                close = float(df['Close'].iloc[i])
                entry_price = pos['entry_price']
                hold_days = i - pos['entry_idx']
                cash += pos['qty'] * close * (1 - commission)
                ret = (close - entry_price) / entry_price * 100 - 2 * commission * 100
                tue_trades.append({'ticker': ticker, 'entry_date': pos['entry_date'], 'exit_date': date, 'ret': ret, 'hold_days': hold_days, 'pnl_usd': pos['qty'] * (close - entry_price)})
                del tue_positions[ticker]

        # ── RSI入场 ──
        for ticker in ['SPY', 'QQQ', 'GLD']:
            if ticker in rsi_positions:
                continue
            df = data[ticker]
            params = rsi_params[ticker]
            close = float(df['Close'].iloc[i])
            rsi2 = float(df['RSI2'].iloc[i])
            sma200 = float(df['SMA200'].iloc[i])
            ibs = float(df['IBS'].iloc[i])

            above_ma = close > sma200
            rsi_low = rsi2 < params['rsi_entry']
            ibs_ok = (not params['use_ibs']) or (ibs < params['ibs_threshold'])

            if above_ma and rsi_low and ibs_ok:
                target = total_equity * params['pos_pct']
                if cash >= target * 0.5:
                    buy_val = min(target, cash * 0.90)
                    qty = max(1, int(buy_val / close))
                    cost = qty * close * (1 + commission)
                    if cost <= cash:
                        cash -= cost
                        rsi_positions[ticker] = {'entry_price': close, 'qty': qty, 'entry_idx': i, 'entry_date': date}

        # ── Tuesday入场(周一) ──
        if weekday == 0:
            for ticker in tue_tickers:
                if ticker in tue_positions: continue
                if ticker in rsi_positions: continue  # 避免叠加
                df = data[ticker]
                close = float(df['Close'].iloc[i])
                prev_close = float(df['prev_close'].iloc[i])
                if pd.isna(prev_close) or prev_close == 0: continue
                drop = (close - prev_close) / prev_close
                if drop <= -tue_drop:
                    target = total_equity * tue_pos_pct
                    if cash >= target * 0.3:
                        buy_val = min(target, cash * 0.90)
                        qty = max(1, int(buy_val / close))
                        cost = qty * close * (1 + commission)
                        if cost <= cash:
                            cash -= cost
                            tue_positions[ticker] = {'entry_price': close, 'qty': qty, 'entry_idx': i, 'entry_date': date}

    # ── 统计 ──
    equity = np.array(equity_curve)
    final = equity[-1]
    total_ret = (final / initial_capital - 1) * 100
    cagr = ((final / initial_capital) ** (1/years) - 1) * 100
    peak = np.maximum.accumulate(equity)
    dd = ((equity - peak) / peak * 100).min()
    daily_rets = np.diff(equity) / equity[:-1]
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else 0

    # SPY buy & hold
    spy_start = float(spy['Close'].iloc[200])
    spy_end = float(spy['Close'].iloc[-1])
    spy_cagr = ((spy_end / spy_start) ** (1/years) - 1) * 100
    spy_eq = initial_capital * (spy['Close'].iloc[200:200+len(equity)].values.astype(float) / spy_start)
    spy_peak = np.maximum.accumulate(spy_eq)
    spy_dd = ((spy_eq - spy_peak) / spy_peak * 100).min()

    rsi_df = pd.DataFrame(rsi_trades) if rsi_trades else pd.DataFrame()
    tue_df = pd.DataFrame(tue_trades) if tue_trades else pd.DataFrame()

    print(f"\n{'='*70}")
    print(f"📊 5策略组合回测 ({years:.1f}年)")
    print(f"{'='*70}")
    print(f"\n  初始: ${initial_capital:,.0f}  →  最终: ${final:,.0f}")
    print(f"  总回报: {total_ret:+.1f}%")
    print(f"  年化CAGR: {cagr:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  最大回撤: {dd:.1f}%")
    print(f"\n  SPY买入持有: CAGR {spy_cagr:.2f}%  回撤 {spy_dd:.1f}%")

    print(f"\n{'─'*70}")
    print(f"RSI均值回归 (SPY+QQQ+GLD):")
    if len(rsi_df):
        for t in ['SPY', 'QQQ', 'GLD']:
            sub = rsi_df[rsi_df['ticker'] == t]
            if len(sub):
                print(f"  {t}: {len(sub)}笔  胜率{(sub['ret']>0).mean()*100:.1f}%  均{sub['ret'].mean():+.3f}%  PnL ${sub['pnl_usd'].sum():+,.0f}")
        print(f"  RSI合计: {len(rsi_df)}笔  总PnL ${rsi_df['pnl_usd'].sum():+,.0f}")

    print(f"\nTurnaround Tuesday (SPY+QQQ):")
    if len(tue_df):
        for t in ['SPY', 'QQQ']:
            sub = tue_df[tue_df['ticker'] == t]
            if len(sub):
                print(f"  {t}: {len(sub)}笔  胜率{(sub['ret']>0).mean()*100:.1f}%  均{sub['ret'].mean():+.3f}%  PnL ${sub['pnl_usd'].sum():+,.0f}")
        print(f"  Tue合计: {len(tue_df)}笔  总PnL ${tue_df['pnl_usd'].sum():+,.0f}")

    total_trades = len(rsi_df) + len(tue_df)
    print(f"\n总交易: {total_trades}笔  年均{total_trades/years:.1f}笔")

    # 年度明细
    print(f"\n{'─'*70}")
    dates_arr = np.array(dates_curve)
    print(f"{'年份':>6s} {'组合':>8s} {'SPY':>8s}")
    print(f"{'─'*70}")
    for year in range(dates_arr[0].year, dates_arr[-1].year + 1):
        mask = np.array([d.year == year for d in dates_arr])
        if not mask.any(): continue
        ye = equity[mask]
        if len(ye) < 2: continue
        yr_ret = (ye[-1] / ye[0] - 1) * 100
        spy_yr = spy_eq[:len(equity)][mask]
        spy_yr_ret = (spy_yr[-1] / spy_yr[0] - 1) * 100 if len(spy_yr) >= 2 else 0
        print(f"  {year}  {yr_ret:>+7.2f}%  {spy_yr_ret:>+7.2f}%")

    print(f"{'─'*70}")
    return {'cagr': round(cagr, 2), 'sharpe': round(sharpe, 2), 'max_dd': round(dd, 1)}


if __name__ == '__main__':
    run()
