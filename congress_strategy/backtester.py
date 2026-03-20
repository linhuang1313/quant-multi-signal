"""
回测系统 - Backtester
======================
使用历史国会交易数据验证策略表现

回测逻辑:
1. 使用历史信号（已评分的买入信号）
2. 在信号发出后的下一个交易日以开盘价买入
3. 持有固定天数后卖出
4. 统计总收益率、胜率、夏普比率等关键指标
5. 对比 Buy & Hold S&P 500 基准
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CongressBacktester:
    """国会跟单策略回测器"""

    def __init__(self,
                 initial_capital: float = 100_000,
                 hold_days: int = 30,
                 max_position_pct: float = 0.10,
                 max_positions: int = 10,
                 stop_loss_pct: float = -0.08,
                 min_signal_score: int = 60):
        """
        Args:
            initial_capital: 初始资金
            hold_days: 持仓天数
            max_position_pct: 单只股票最大仓位占比
            max_positions: 最大同时持仓数
            stop_loss_pct: 止损比例 (-0.08 = -8%)
            min_signal_score: 最低信号评分
        """
        self.initial_capital = initial_capital
        self.hold_days = hold_days
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.min_signal_score = min_signal_score
        
        # 价格缓存
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def run_backtest(self, signals_df: pd.DataFrame,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict:
        """
        运行策略回测
        
        Args:
            signals_df: 已评分的信号 DataFrame (必须包含 total_score 列)
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
            
        Returns:
            dict: 回测结果 (收益率、胜率、交易记录等)
        """
        # 过滤信号
        df = signals_df[
            (signals_df['trade_type'] == 'Purchase') &
            (signals_df['total_score'] >= self.min_signal_score)
        ].copy()
        
        if start_date:
            df = df[df['transaction_date'] >= start_date]
        if end_date:
            df = df[df['transaction_date'] <= end_date]
        
        df = df.sort_values('transaction_date').reset_index(drop=True)
        
        if df.empty:
            print("⚠️ 没有符合条件的信号用于回测")
            return self._empty_result()
        
        print(f"📊 回测参数:")
        print(f"  初始资金:    ${self.initial_capital:,.0f}")
        print(f"  持仓天数:    {self.hold_days} 天")
        print(f"  最大仓位:    {self.max_position_pct*100:.0f}%")
        print(f"  止损线:      {self.stop_loss_pct*100:.0f}%")
        print(f"  信号数量:    {len(df)}")
        print(f"  回测区间:    {df['transaction_date'].min():%Y-%m-%d} 至 {df['transaction_date'].max():%Y-%m-%d}")
        print()
        
        # 预加载价格数据
        tickers = df['ticker'].unique().tolist() + ['SPY']
        print(f"📥 加载 {len(tickers)} 只股票的价格数据...")
        self._preload_prices(tickers, 
                            start=df['transaction_date'].min() - timedelta(days=5),
                            end=datetime.now())
        
        # 模拟交易
        trades = self._simulate_trades(df)
        
        if not trades:
            print("⚠️ 没有成功执行的交易")
            return self._empty_result()
        
        # 计算统计指标
        result = self._calculate_metrics(trades)
        
        # 计算基准收益
        result['benchmark'] = self._calculate_benchmark(
            df['transaction_date'].min(), 
            df['transaction_date'].max()
        )
        
        return result

    def _preload_prices(self, tickers: List[str], 
                        start: datetime, end: datetime):
        """批量预加载价格数据"""
        missing = [t for t in tickers if t not in self._price_cache]
        if not missing:
            return
        
        # 分批下载避免超时
        batch_size = 20
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i+batch_size]
            try:
                data = yf.download(
                    batch, 
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    progress=False,
                    group_by='ticker' if len(batch) > 1 else 'column'
                )
                
                if len(batch) == 1:
                    self._price_cache[batch[0]] = data
                else:
                    for ticker in batch:
                        if ticker in data.columns.get_level_values(0):
                            self._price_cache[ticker] = data[ticker]
            except Exception as e:
                logger.warning(f"批量下载失败: {e}")
                # 逐个下载
                for ticker in batch:
                    try:
                        self._price_cache[ticker] = yf.download(
                            ticker, start=start.strftime('%Y-%m-%d'),
                            end=end.strftime('%Y-%m-%d'), progress=False
                        )
                    except:
                        pass

    def _get_price(self, ticker: str, date: datetime, 
                   offset_days: int = 0) -> Optional[float]:
        """获取某只股票在某日期的收盘价"""
        if ticker not in self._price_cache:
            return None
        
        prices = self._price_cache[ticker]
        if prices.empty:
            return None
        
        target_date = date + timedelta(days=offset_days)
        
        # 找到最近的交易日
        try:
            # Handle both MultiIndex and regular Index
            if isinstance(prices.columns, pd.MultiIndex):
                close_col = prices[('Close', ticker)] if ('Close', ticker) in prices.columns else prices['Close']
            else:
                close_col = prices['Close']
            
            available = close_col.dropna()
            if available.empty:
                return None
            
            # 向前查找最近的交易日 (最多找 5 天)
            for d in range(6):
                check_date = target_date + timedelta(days=d)
                check_str = check_date.strftime('%Y-%m-%d')
                if check_str in available.index.strftime('%Y-%m-%d').values:
                    idx = available.index[available.index.strftime('%Y-%m-%d') == check_str][0]
                    val = available[idx]
                    return float(val) if not pd.isna(val) else None
            return None
        except Exception:
            return None

    def _simulate_trades(self, signals: pd.DataFrame) -> List[Dict]:
        """模拟交易执行"""
        trades = []
        capital = self.initial_capital
        active_positions = {}  # ticker -> {entry_price, shares, entry_date}
        
        for _, signal in signals.iterrows():
            ticker = signal['ticker']
            tx_date = signal['transaction_date']
            
            if pd.isna(tx_date):
                continue
            
            # 检查是否已持有该股票
            if ticker in active_positions:
                continue
            
            # 检查持仓数量限制
            if len(active_positions) >= self.max_positions:
                # 先平掉到期的仓位
                self._close_expired(active_positions, trades, tx_date, capital)
            
            if len(active_positions) >= self.max_positions:
                continue
            
            # 计算仓位大小
            position_size = capital * self.max_position_pct
            
            # 获取买入价格 (信号后第一个交易日开盘价)
            entry_price = self._get_price(ticker, tx_date, offset_days=1)
            if entry_price is None or entry_price <= 0:
                continue
            
            shares = int(position_size / entry_price)
            if shares <= 0:
                continue
            
            actual_cost = shares * entry_price
            
            # 获取卖出价格 (持仓期结束后的收盘价)
            exit_date = tx_date + timedelta(days=self.hold_days)
            exit_price = self._get_price(ticker, exit_date)
            
            # 如果无法获取到退出价格，尝试用最新价
            if exit_price is None:
                exit_price = self._get_price(ticker, datetime.now() - timedelta(days=5))
            
            if exit_price is None:
                continue
            
            # 检查止损
            return_pct = (exit_price - entry_price) / entry_price
            stopped_out = return_pct <= self.stop_loss_pct
            
            if stopped_out:
                exit_price = entry_price * (1 + self.stop_loss_pct)
            
            profit = (exit_price - entry_price) * shares
            
            trade = {
                'ticker': ticker,
                'representative': signal['representative'],
                'signal_score': signal['total_score'],
                'entry_date': tx_date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'shares': shares,
                'cost': actual_cost,
                'profit': profit,
                'return_pct': return_pct * 100,
                'stopped_out': stopped_out,
                'amount_range': signal['amount'],
            }
            trades.append(trade)
            
            # 更新资本
            capital += profit
        
        return trades

    def _close_expired(self, positions: Dict, trades: List, 
                       current_date: datetime, capital: float):
        """平掉到期的仓位"""
        to_close = []
        for ticker, pos in positions.items():
            if current_date >= pos['entry_date'] + timedelta(days=self.hold_days):
                to_close.append(ticker)
        
        for ticker in to_close:
            del positions[ticker]

    def _calculate_metrics(self, trades: List[Dict]) -> Dict:
        """计算回测指标"""
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        winners = trades_df[trades_df['profit'] > 0]
        losers = trades_df[trades_df['profit'] <= 0]
        
        total_profit = trades_df['profit'].sum()
        total_return = total_profit / self.initial_capital * 100
        
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = winners['return_pct'].mean() if len(winners) > 0 else 0
        avg_loss = losers['return_pct'].mean() if len(losers) > 0 else 0
        
        # 收益序列
        returns = trades_df['return_pct'] / 100
        sharpe = (returns.mean() / returns.std() * np.sqrt(252/self.hold_days)) if returns.std() > 0 else 0
        
        max_drawdown = trades_df['return_pct'].min()
        
        # 按议员分组统计
        by_rep = trades_df.groupby('representative').agg({
            'profit': 'sum',
            'return_pct': 'mean',
            'ticker': 'count'
        }).rename(columns={'ticker': 'trade_count'})
        by_rep = by_rep.sort_values('profit', ascending=False)
        
        result = {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'total_return_pct': total_return,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'sharpe_ratio': sharpe,
            'max_single_loss_pct': max_drawdown,
            'stopped_out_count': trades_df['stopped_out'].sum(),
            'trades_df': trades_df,
            'by_representative': by_rep,
            'final_capital': self.initial_capital + total_profit,
        }
        
        return result

    def _calculate_benchmark(self, start_date: datetime, 
                             end_date: datetime) -> Dict:
        """计算 S&P 500 基准收益"""
        try:
            spy_start = self._get_price('SPY', start_date)
            spy_end = self._get_price('SPY', end_date)
            
            if spy_start and spy_end:
                return {
                    'ticker': 'SPY',
                    'return_pct': (spy_end - spy_start) / spy_start * 100,
                    'start_price': spy_start,
                    'end_price': spy_end,
                }
        except:
            pass
        
        return {'ticker': 'SPY', 'return_pct': 0, 'start_price': 0, 'end_price': 0}

    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'total_trades': 0,
            'total_profit': 0,
            'total_return_pct': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'trades_df': pd.DataFrame(),
            'benchmark': {'return_pct': 0},
        }

    @staticmethod
    def format_result(result: Dict) -> str:
        """格式化回测结果"""
        if result['total_trades'] == 0:
            return "❌ 无回测数据"
        
        lines = [
            "",
            "=" * 65,
            "📊 国会跟单策略 — 回测报告",
            "=" * 65,
            "",
            "📈 收益概览:",
            f"  初始资金:      ${100_000:>12,.0f}",
            f"  最终资金:      ${result['final_capital']:>12,.2f}",
            f"  总收益:        ${result['total_profit']:>12,.2f}",
            f"  总收益率:       {result['total_return_pct']:>10.2f}%",
            f"  SPY 基准收益:   {result['benchmark']['return_pct']:>10.2f}%",
            f"  超额收益(Alpha):{result['total_return_pct'] - result['benchmark']['return_pct']:>10.2f}%",
            "",
            "📋 交易统计:",
            f"  总交易笔数:    {result['total_trades']:>6d}",
            f"  胜率:          {result['win_rate']:>10.1f}%",
            f"  平均盈利:      {result.get('avg_win_pct', 0):>10.2f}%",
            f"  平均亏损:      {result.get('avg_loss_pct', 0):>10.2f}%",
            f"  夏普比率:      {result['sharpe_ratio']:>10.2f}",
            f"  最大单笔亏损:  {result.get('max_single_loss_pct', 0):>10.2f}%",
            f"  触发止损次数:  {result.get('stopped_out_count', 0):>6d}",
        ]
        
        # 最佳/最差交易
        if not result['trades_df'].empty:
            best = result['trades_df'].loc[result['trades_df']['return_pct'].idxmax()]
            worst = result['trades_df'].loc[result['trades_df']['return_pct'].idxmin()]
            
            lines.extend([
                "",
                "🏆 最佳交易:",
                f"  {best['ticker']} | +{best['return_pct']:.2f}% | "
                f"${best['profit']:,.2f} | 跟随 {best['representative']}",
                "",
                "💔 最差交易:",
                f"  {worst['ticker']} | {worst['return_pct']:.2f}% | "
                f"${worst['profit']:,.2f} | 跟随 {worst['representative']}",
            ])
            
            # 按议员排行
            if 'by_representative' in result:
                lines.extend([
                    "",
                    "🏛️ 议员交易表现 Top 5:",
                    f"  {'议员':25s} {'笔数':>5s} {'平均收益':>8s} {'总利润':>12s}",
                    "  " + "-" * 55,
                ])
                for name, row in result['by_representative'].head(5).iterrows():
                    lines.append(
                        f"  {name[:25]:25s} {row['trade_count']:>5.0f} "
                        f"{row['return_pct']:>7.2f}% ${row['profit']:>11,.2f}"
                    )
        
        lines.extend(["", "=" * 65])
        return "\n".join(lines)


# ============================================================
# 便捷函数
# ============================================================

def run_backtest(signals_df: pd.DataFrame, **kwargs) -> Dict:
    """
    一键运行回测
    
    Args:
        signals_df: 已评分的信号数据
        **kwargs: 传递给 CongressBacktester 的参数
        
    Returns:
        dict: 回测结果
    """
    bt = CongressBacktester(**kwargs)
    return bt.run_backtest(signals_df)


if __name__ == "__main__":
    from data_fetcher import fetch_congress_trades
    from signal_scorer import SignalScorer
    
    # 获取数据
    df = fetch_congress_trades()
    
    # 评分
    scorer = SignalScorer(min_score=50)
    scored = scorer.score_trades(df)
    
    # 回测
    bt = CongressBacktester(hold_days=30, min_signal_score=50)
    result = bt.run_backtest(scored)
    print(CongressBacktester.format_result(result))
