"""
参数优化回测 - Parameter Optimization Backtest
================================================
对历史共振信号进行回测，寻找最优参数组合

测试维度:
- 持仓天数: 10, 15, 20, 30, 45
- 止盈: 8%, 12%, 15%, 20%, 25%
- 止损: 5%, 8%, 10%, 12%
- 移动止损: 4%, 6%, 8% (从高点回撤)

评估指标:
- 总收益率
- 胜率
- 平均收益/平均亏损 (盈亏比)
- 最大回撤
- Sharpe Ratio (简化版)
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

# ============================================================
# 回测参数网格
# ============================================================
HOLD_DAYS_LIST = [10, 15, 20, 30, 45]
TAKE_PROFIT_LIST = [0.08, 0.12, 0.15, 0.20, 0.25]
STOP_LOSS_LIST = [0.05, 0.08, 0.10, 0.12]
TRAILING_STOP_LIST = [0.04, 0.06, 0.08]


def get_historical_signals() -> pd.DataFrame:
    """
    从历史数据中生成模拟的共振信号
    使用过去 12 个月的国会+内部人数据回溯
    """
    from data_fetcher import CongressDataFetcher
    from insider_fetcher import InsiderFetcher

    print("📥 获取历史数据...")
    
    # 获取全部国会数据
    fetcher = CongressDataFetcher()
    congress_all = fetcher.fetch_all(use_cache=True)
    
    # 获取内部人数据
    insider_fetcher = InsiderFetcher()
    insider_all = insider_fetcher.fetch_all(use_cache=True)
    
    # 从国会数据中提取买入信号 (最近 12 个月)
    cutoff = datetime.now() - timedelta(days=365)
    
    signals = []
    
    # 国会买入信号
    if not congress_all.empty:
        congress_buys = congress_all[
            (congress_all['trade_type'].str.contains('Purchase', case=False, na=False)) &
            (congress_all['transaction_date'] >= cutoff)
        ].copy()
        
        for _, row in congress_buys.iterrows():
            signals.append({
                'ticker': row['ticker'],
                'signal_date': row['transaction_date'],
                'source': 'congress',
                'representative': row.get('representative', ''),
                'amount': row.get('amount_est', 8000),
            })
    
    # 内部人买入信号
    if not insider_all.empty:
        for _, row in insider_all.iterrows():
            td = row.get('trade_date')
            if pd.notna(td) and pd.Timestamp(td) >= pd.Timestamp(cutoff):
                signals.append({
                    'ticker': row['ticker'],
                    'signal_date': pd.Timestamp(td),
                    'source': 'insider',
                    'insider_name': row.get('insider_name', ''),
                    'value': row.get('value', 0),
                })
    
    signals_df = pd.DataFrame(signals)
    if signals_df.empty:
        return signals_df
        
    signals_df['signal_date'] = pd.to_datetime(signals_df['signal_date'])
    
    # 找出多信号重叠的 ticker (同一周内多个信号源指向同一 ticker)
    # 模拟共振检测
    signals_df = signals_df.sort_values('signal_date')
    
    print(f"📊 历史信号: {len(signals_df)} 个 ({signals_df['ticker'].nunique()} 只标的)")
    print(f"   国会: {len(signals_df[signals_df['source']=='congress'])} 个")
    print(f"   内部人: {len(signals_df[signals_df['source']=='insider'])} 个")
    
    return signals_df


def detect_historical_resonance(signals_df: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    """
    检测历史上的信号共振 (同一 ticker 在 window_days 内出现多个不同来源的信号)
    """
    if signals_df.empty:
        return pd.DataFrame()
    
    resonance_signals = []
    
    # 按 ticker 分组
    for ticker, group in signals_df.groupby('ticker'):
        sources = group['source'].unique()
        
        if len(sources) >= 2:
            # 该 ticker 有多个信号源 — 检查时间窗口
            dates = group['signal_date'].sort_values()
            # 取最新的信号日期作为入场点
            entry_date = dates.iloc[-1]
            
            resonance_signals.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'resonance_level': len(sources),
                'sources': '+'.join(sorted(sources)),
                'signal_count': len(group),
            })
        else:
            # 单信号 — 也纳入回测对比
            best = group.sort_values('signal_date').iloc[-1]
            resonance_signals.append({
                'ticker': ticker,
                'entry_date': best['signal_date'],
                'resonance_level': 1,
                'sources': sources[0],
                'signal_count': len(group),
            })
    
    result = pd.DataFrame(resonance_signals)
    
    if not result.empty:
        r_counts = result['resonance_level'].value_counts().to_dict()
        print(f"\n📊 历史共振分布:")
        print(f"   双重/三重共振: {r_counts.get(2,0) + r_counts.get(3,0)} 只")
        print(f"   单信号: {r_counts.get(1,0)} 只")
    
    return result


def fetch_price_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """批量获取价格数据"""
    print(f"\n📥 获取 {len(tickers)} 只标的的价格数据...")
    
    price_data = {}
    batch_size = 20
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        tickers_str = ' '.join(batch)
        
        try:
            data = yf.download(tickers_str, start=start_date, end=end_date, 
                             progress=False, auto_adjust=True)
            
            if len(batch) == 1:
                # 单只标的
                if not data.empty:
                    price_data[batch[0]] = data
            else:
                # 多只标的
                if 'Close' in data.columns.names or hasattr(data.columns, 'levels'):
                    for ticker in batch:
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                ticker_data = data.xs(ticker, axis=1, level=1) if ticker in data.columns.get_level_values(1) else None
                            else:
                                ticker_data = data
                            if ticker_data is not None and not ticker_data.empty:
                                price_data[ticker] = ticker_data
                        except (KeyError, Exception):
                            continue
                else:
                    if not data.empty:
                        price_data[batch[0]] = data
        except Exception as e:
            print(f"  ⚠️ 批次获取失败: {e}")
            # 逐只获取
            for ticker in batch:
                try:
                    data = yf.download(ticker, start=start_date, end=end_date, 
                                     progress=False, auto_adjust=True)
                    if not data.empty:
                        price_data[ticker] = data
                except Exception:
                    continue
    
    print(f"  获取到 {len(price_data)} 只标的的价格数据")
    return price_data


def simulate_trade(price_df: pd.DataFrame, entry_date: pd.Timestamp,
                   max_hold_days: int, take_profit: float, stop_loss: float,
                   trailing_stop: float) -> Dict:
    """
    模拟单笔交易
    
    Returns:
        Dict: 交易结果 (return_pct, hold_days, exit_reason)
    """
    if price_df is None or price_df.empty:
        return None
    
    # 找到入场日期之后的第一个交易日
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    
    # 确保 entry_date 是 tz-naive
    if hasattr(entry_date, 'tz') and entry_date.tz is not None:
        entry_date = entry_date.tz_localize(None)
    if price_df.index.tz is not None:
        price_df.index = price_df.index.tz_localize(None)
    
    future = price_df[price_df.index >= entry_date]
    if len(future) < 2:
        return None
    
    # 获取 Close 列
    if isinstance(future.columns, pd.MultiIndex):
        close = future['Close'].iloc[:, 0] if 'Close' in future.columns.get_level_values(0) else None
        high = future['High'].iloc[:, 0] if 'High' in future.columns.get_level_values(0) else None
    elif 'Close' in future.columns:
        close = future['Close']
        high = future.get('High', future['Close'])
    else:
        return None
    
    if close is None or close.empty:
        return None
    
    entry_price = float(close.iloc[0])
    if entry_price <= 0:
        return None
    
    # 限制到最大持仓天数
    max_bars = min(len(close), max_hold_days + 1)
    
    highest_price = entry_price
    
    for i in range(1, max_bars):
        current_price = float(close.iloc[i])
        if high is not None:
            day_high = float(high.iloc[i])
            highest_price = max(highest_price, day_high)
        else:
            highest_price = max(highest_price, current_price)
        
        return_pct = (current_price - entry_price) / entry_price
        drawdown_from_high = (current_price - highest_price) / highest_price if highest_price > 0 else 0
        
        # 止盈
        if return_pct >= take_profit:
            return {
                'return_pct': return_pct,
                'hold_days': i,
                'exit_reason': 'take_profit',
                'exit_price': current_price,
            }
        
        # 止损
        if return_pct <= -stop_loss:
            return {
                'return_pct': return_pct,
                'hold_days': i,
                'exit_reason': 'stop_loss',
                'exit_price': current_price,
            }
        
        # 移动止损 (盈利超过 3% 才启用)
        if return_pct > 0.03 and drawdown_from_high <= -trailing_stop:
            return {
                'return_pct': return_pct,
                'hold_days': i,
                'exit_reason': 'trailing_stop',
                'exit_price': current_price,
            }
    
    # 到期平仓
    final_price = float(close.iloc[max_bars - 1])
    return_pct = (final_price - entry_price) / entry_price
    return {
        'return_pct': return_pct,
        'hold_days': max_bars - 1,
        'exit_reason': 'max_hold',
        'exit_price': final_price,
    }


def run_param_backtest(signals: pd.DataFrame, price_data: Dict,
                       hold_days: int, take_profit: float, 
                       stop_loss: float, trailing_stop: float) -> Dict:
    """对一组参数运行完整回测"""
    
    trades = []
    
    for _, sig in signals.iterrows():
        ticker = sig['ticker']
        entry_date = sig['entry_date']
        
        if ticker not in price_data:
            continue
        
        result = simulate_trade(
            price_data[ticker], entry_date,
            hold_days, take_profit, stop_loss, trailing_stop
        )
        
        if result:
            result['ticker'] = ticker
            result['resonance_level'] = sig['resonance_level']
            result['entry_date'] = entry_date
            trades.append(result)
    
    if not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    
    # 计算指标
    returns = trades_df['return_pct']
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    
    total_return = (1 + returns).prod() - 1  # 复合收益
    avg_return = returns.mean()
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.001
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # 简化 Sharpe (假设无风险利率 0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252/hold_days) if returns.std() > 0 else 0
    
    # 最大回撤 (按交易序列)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # 退出原因分布
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
    
    # 按共振等级分析
    resonance_returns = trades_df.groupby('resonance_level')['return_pct'].agg(['mean', 'count']).to_dict('index')
    
    return {
        'hold_days': hold_days,
        'take_profit': take_profit,
        'stop_loss': stop_loss,
        'trailing_stop': trailing_stop,
        'total_trades': len(trades),
        'total_return': round(total_return * 100, 2),
        'avg_return': round(avg_return * 100, 2),
        'win_rate': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_drawdown * 100, 2),
        'avg_hold_days': round(trades_df['hold_days'].mean(), 1),
        'exit_take_profit': exit_reasons.get('take_profit', 0),
        'exit_stop_loss': exit_reasons.get('stop_loss', 0),
        'exit_trailing_stop': exit_reasons.get('trailing_stop', 0),
        'exit_max_hold': exit_reasons.get('max_hold', 0),
        'resonance_returns': resonance_returns,
    }


def main():
    print("=" * 70)
    print("🧪 参数优化回测")
    print("=" * 70)
    
    # 1. 获取历史信号
    signals_df = get_historical_signals()
    if signals_df.empty:
        print("❌ 无历史信号数据")
        return
    
    # 2. 检测历史共振
    resonance = detect_historical_resonance(signals_df)
    if resonance.empty:
        print("❌ 无共振信号")
        return
    
    # 3. 获取价格数据
    tickers = resonance['ticker'].unique().tolist()
    start_date = (resonance['entry_date'].min() - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    price_data = fetch_price_data(tickers, start_date, end_date)
    
    # 4. 参数网格回测
    param_combos = list(product(
        HOLD_DAYS_LIST, TAKE_PROFIT_LIST, STOP_LOSS_LIST, TRAILING_STOP_LIST
    ))
    
    print(f"\n🔄 测试 {len(param_combos)} 种参数组合...")
    
    all_results = []
    for i, (hd, tp, sl, ts) in enumerate(param_combos):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(param_combos)}...")
        
        result = run_param_backtest(resonance, price_data, hd, tp, sl, ts)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("❌ 回测无结果")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # 5. 输出结果
    print(f"\n{'='*70}")
    print(f"📊 回测结果 ({len(results_df)} 种有效参数组合)")
    print(f"{'='*70}")
    
    # 按 Sharpe 排序
    results_df = results_df.sort_values('sharpe', ascending=False)
    
    # Top 10 参数组合
    print(f"\n🏆 Top 10 最佳参数组合 (按 Sharpe Ratio 排序):")
    print(f"{'持仓':>4s}  {'止盈':>5s}  {'止损':>5s}  {'移动':>5s}  {'总收益':>7s}  "
          f"{'胜率':>5s}  {'盈亏比':>6s}  {'Sharpe':>7s}  {'最大回撤':>8s}  {'交易数':>5s}")
    print("-" * 80)
    
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['hold_days']:>2.0f}天  {row['take_profit']*100:>4.0f}%  "
              f"{row['stop_loss']*100:>4.0f}%  {row['trailing_stop']*100:>4.0f}%  "
              f"{row['total_return']:>+6.1f}%  {row['win_rate']:>4.1f}%  "
              f"{row['profit_factor']:>5.2f}  {row['sharpe']:>6.2f}  "
              f"{row['max_drawdown']:>+7.1f}%  {row['total_trades']:>4.0f}")
    
    # 最差 5 组 (对比)
    print(f"\n❌ 最差 5 组参数:")
    for _, row in results_df.tail(5).iterrows():
        print(f"  {row['hold_days']:>2.0f}天  {row['take_profit']*100:>4.0f}%  "
              f"{row['stop_loss']*100:>4.0f}%  {row['trailing_stop']*100:>4.0f}%  "
              f"{row['total_return']:>+6.1f}%  {row['win_rate']:>4.1f}%  "
              f"{row['sharpe']:>6.2f}")
    
    # 按持仓天数聚合分析
    print(f"\n📊 按持仓天数汇总:")
    hold_agg = results_df.groupby('hold_days').agg({
        'total_return': 'mean',
        'win_rate': 'mean',
        'sharpe': 'mean',
    }).round(2)
    for hd, row in hold_agg.iterrows():
        print(f"  {hd:>2.0f}天: 平均收益 {row['total_return']:>+6.1f}%  "
              f"平均胜率 {row['win_rate']:>4.1f}%  平均Sharpe {row['sharpe']:>5.2f}")
    
    # 按止盈聚合
    print(f"\n📊 按止盈比例汇总:")
    tp_agg = results_df.groupby('take_profit').agg({
        'total_return': 'mean',
        'win_rate': 'mean',
        'sharpe': 'mean',
    }).round(2)
    for tp, row in tp_agg.iterrows():
        print(f"  {tp*100:>4.0f}%: 平均收益 {row['total_return']:>+6.1f}%  "
              f"平均胜率 {row['win_rate']:>4.1f}%  平均Sharpe {row['sharpe']:>5.2f}")
    
    # 按止损聚合
    print(f"\n📊 按止损比例汇总:")
    sl_agg = results_df.groupby('stop_loss').agg({
        'total_return': 'mean',
        'win_rate': 'mean',
        'sharpe': 'mean',
    }).round(2)
    for sl, row in sl_agg.iterrows():
        print(f"  {sl*100:>4.0f}%: 平均收益 {row['total_return']:>+6.1f}%  "
              f"平均胜率 {row['win_rate']:>4.1f}%  平均Sharpe {row['sharpe']:>5.2f}")
    
    # 最优参数
    best = results_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"🏆 最优参数组合:")
    print(f"  持仓天数: {best['hold_days']:.0f} 天")
    print(f"  止盈: {best['take_profit']*100:.0f}%")
    print(f"  止损: {best['stop_loss']*100:.0f}%")
    print(f"  移动止损: {best['trailing_stop']*100:.0f}%")
    print(f"  ---")
    print(f"  总收益: {best['total_return']:+.1f}%")
    print(f"  胜率: {best['win_rate']:.1f}%")
    print(f"  盈亏比: {best['profit_factor']:.2f}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  最大回撤: {best['max_drawdown']:.1f}%")
    print(f"  交易次数: {best['total_trades']:.0f}")
    print(f"{'='*70}")
    
    # 保存结果
    output_file = __import__('pathlib').Path(__file__).parent / "data" / "param_backtest_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 结果已保存: {output_file}")
    
    return results_df


if __name__ == "__main__":
    results = main()
