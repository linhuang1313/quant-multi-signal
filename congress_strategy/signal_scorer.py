"""
信号评分引擎 v2 - Signal Scorer
================================
对每笔国会交易进行多维度评分，筛选出高置信度的跟单信号

v2 改进:
- 重新校准评分权重，确保各维度有足够区分度
- 新增"议员活跃度"和"股票流动性"评分维度
- 集群检测窗口从 7 天扩大到 14 天
- 对缺失字段的数据用合理替代方案评分

评分维度 (总分 100):
1. 交易金额 (0-20分): 大额交易 = 强信号
2. 委员会相关性 (0-15分): 议员 x 行业匹配度
3. 集群信号 (0-25分): 多位议员短时间内买同一股票
4. 申报速度 (0-10分): 快速申报的交易更值得关注
5. 议员画像 (0-15分): 交易频率、胜率、知名度
6. 股票质量 (0-15分): 大盘蓝筹 > 冷门小票
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================
# 委员会 → 行业映射
# ============================================================

NOTABLE_MEMBERS = {
    'Nancy Pelosi': {'sectors': ['Technology', 'Communication Services'], 'fame': 15},
    'Dan Crenshaw': {'sectors': ['Energy', 'Technology'], 'fame': 10},
    'Tommy Tuberville': {'sectors': ['Financials', 'Industrials'], 'fame': 10},
    'Mitch McConnell': {'sectors': ['Financials', 'Industrials'], 'fame': 12},
    'Mark Kelly': {'sectors': ['Industrials', 'Technology'], 'fame': 8},
    'Thomas H. Kean': {'sectors': ['Financials', 'Technology'], 'fame': 7},
    'Marjorie Taylor Greene': {'sectors': ['Technology', 'Communication Services'], 'fame': 10},
    'Josh Gottheimer': {'sectors': ['Financials', 'Technology'], 'fame': 8},
    'Michael McCaul': {'sectors': ['Technology', 'Industrials'], 'fame': 10},
    'Michael T. McCaul': {'sectors': ['Technology', 'Industrials'], 'fame': 10},
    'Ro Khanna': {'sectors': ['Technology'], 'fame': 9},
    'Kelly Loeffler': {'sectors': ['Financials', 'Technology'], 'fame': 12},
    'David Perdue': {'sectors': ['Financials', 'Technology', 'Consumer Discretionary'], 'fame': 10},
    'Richard Burr': {'sectors': ['Health Care', 'Industrials'], 'fame': 10},
    'Dianne Feinstein': {'sectors': ['Technology', 'Real Estate'], 'fame': 12},
    'Sheldon Whitehouse': {'sectors': ['Energy', 'Financials'], 'fame': 7},
    'Pat Roberts': {'sectors': ['Consumer Staples', 'Industrials'], 'fame': 6},
    'Susan Collins': {'sectors': ['Health Care', 'Financials'], 'fame': 8},
    'Ron Wyden': {'sectors': ['Technology', 'Financials'], 'fame': 8},
    'John Hoeven': {'sectors': ['Energy', 'Financials'], 'fame': 6},
    'Shelley Capito': {'sectors': ['Energy', 'Financials'], 'fame': 6},
    'Thomas Carper': {'sectors': ['Financials', 'Technology'], 'fame': 6},
    'James Inhofe': {'sectors': ['Energy', 'Industrials'], 'fame': 7},
    'April McClain Delaney': {'sectors': ['Financials', 'Technology'], 'fame': 5},
    'Julia Letlow': {'sectors': ['Health Care', 'Energy'], 'fame': 5},
    'Cleo Fields': {'sectors': ['Technology', 'Communication Services'], 'fame': 6},
    'Markwayne Mullin': {'sectors': ['Energy', 'Industrials'], 'fame': 7},
}

# 股票 → 行业映射 (覆盖更多 ticker)
TICKER_SECTOR = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology', 'CRM': 'Technology',
    'AVGO': 'Technology', 'CSCO': 'Technology', 'ORCL': 'Technology', 'ADBE': 'Technology',
    'QCOM': 'Technology', 'TXN': 'Technology', 'MU': 'Technology', 'AMAT': 'Technology',
    'NOW': 'Technology', 'PANW': 'Technology', 'SNPS': 'Technology', 'CDNS': 'Technology',
    'INTU': 'Technology', 'ADSK': 'Technology', 'WDAY': 'Technology', 'MORN': 'Technology',
    'PAYX': 'Technology', 'ENTG': 'Technology', 'MIDD': 'Technology', 'FCN': 'Technology',
    'HURN': 'Technology',
    # Communication Services
    'META': 'Communication Services', 'NFLX': 'Communication Services',
    'DIS': 'Communication Services', 'CMCSA': 'Communication Services',
    'WBD': 'Communication Services', 'PARA': 'Communication Services',
    'DISCA': 'Communication Services', 'DISCB': 'Communication Services',
    'T': 'Communication Services', 'VZ': 'Communication Services',
    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'MS': 'Financials', 'C': 'Financials', 'BLK': 'Financials', 'SCHW': 'Financials',
    'AXP': 'Financials', 'V': 'Financials', 'MA': 'Financials', 'PYPL': 'Financials',
    'COF': 'Financials', 'USB': 'Financials', 'PNC': 'Financials',
    'ACI': 'Financials', 'JLL': 'Financials', 'AJG': 'Financials',
    # Health Care
    'UNH': 'Health Care', 'JNJ': 'Health Care', 'PFE': 'Health Care', 'ABBV': 'Health Care',
    'MRK': 'Health Care', 'LLY': 'Health Care', 'TMO': 'Health Care', 'ABT': 'Health Care',
    'BMY': 'Health Care', 'AMGN': 'Health Care', 'GILD': 'Health Care', 'MRNA': 'Health Care',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'OXY': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy',
    'VLO': 'Energy', 'HAL': 'Energy',
    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'HD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'TGT': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
    'WWD': 'Consumer Discretionary',
    # Industrials
    'BA': 'Industrials', 'CAT': 'Industrials', 'HON': 'Industrials',
    'UPS': 'Industrials', 'RTX': 'Industrials', 'LMT': 'Industrials',
    'GE': 'Industrials', 'DE': 'Industrials', 'NOC': 'Industrials',
    'GD': 'Industrials', 'LHX': 'Industrials', 'CMI': 'Industrials',
    'EME': 'Industrials', 'BJ': 'Industrials',
    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    # Utilities & Real Estate
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate',
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'ECL': 'Materials', 'SHW': 'Materials',
    'NEM': 'Materials', 'FCX': 'Materials',
}

# 蓝筹股列表 (S&P 100 级别的高流动性大盘股)
BLUE_CHIPS = {
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA',
    'JPM', 'V', 'UNH', 'JNJ', 'HD', 'PG', 'MA', 'BAC', 'XOM', 'CVX',
    'ABBV', 'MRK', 'PFE', 'KO', 'PEP', 'WMT', 'COST', 'DIS', 'CSCO',
    'CRM', 'AMD', 'INTC', 'AVGO', 'QCOM', 'TXN', 'NFLX', 'ADBE',
    'LLY', 'ABT', 'TMO', 'BMY', 'AMGN', 'GILD',
    'BA', 'CAT', 'HON', 'GE', 'RTX', 'LMT', 'NOC',
    'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW',
    'NEE', 'DUK', 'SO',
    'LIN', 'SHW', 'NKE', 'MCD', 'SBUX', 'LOW', 'TGT',
}

# 中盘知名股 (流动性尚可)
MID_CAPS = {
    'INTU', 'ADSK', 'WDAY', 'NOW', 'PANW', 'SNPS', 'CDNS',
    'PAYX', 'MORN', 'CMI', 'EME', 'JLL', 'AJG',
    'MRNA', 'OXY', 'SLB', 'COP', 'EOG', 'MPC', 'PSX', 'VLO', 'HAL',
    'COF', 'USB', 'PNC', 'PYPL',
    'UPS', 'DE', 'GD', 'LHX',
    'WBD', 'PARA', 'CMCSA', 'T', 'VZ',
    'APD', 'ECL', 'NEM', 'FCX',
    'AMT', 'PLD', 'CCI',
}


class SignalScorer:
    """国会交易信号评分器 v2"""

    def __init__(self, 
                 min_score: int = 60,
                 cluster_window_days: int = 14):
        """
        Args:
            min_score: 最低信号分数 (低于此分数不生成信号)
            cluster_window_days: 集群买入检测窗口 (天)
        """
        self.min_score = min_score
        self.cluster_window_days = cluster_window_days
        
        # 预计算议员历史统计 (在 score_trades 中填充)
        self._rep_stats: Dict[str, Dict] = {}

    def score_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对所有交易进行评分
        """
        if df.empty:
            return df
        
        scored = df.copy()
        
        # 预计算议员统计数据
        self._compute_rep_stats(scored)
        
        # 仅对 Purchase 类型评分 (跟单买入策略)
        buy_mask = scored['trade_type'] == 'Purchase'
        
        # 1. 金额评分 (0-20)
        scored['score_amount'] = scored['amount'].apply(self._score_amount)
        
        # 2. 委员会相关性评分 (0-15)
        scored['score_committee'] = scored.apply(
            lambda row: self._score_committee(row['representative'], row['ticker']), axis=1
        )
        
        # 3. 集群信号评分 (0-25)
        scored['score_cluster'] = self._score_cluster_batch(scored)
        
        # 4. 申报速度评分 (0-10)
        scored['score_filing_speed'] = scored['filing_delay_days'].apply(self._score_filing_speed)
        
        # 5. 议员画像评分 (0-15)
        scored['score_rep_profile'] = scored['representative'].apply(self._score_rep_profile)
        
        # 6. 股票质量评分 (0-15)
        scored['score_stock_quality'] = scored['ticker'].apply(self._score_stock_quality)
        
        # 总分
        scored['total_score'] = (
            scored['score_amount'] + 
            scored['score_committee'] + 
            scored['score_cluster'] + 
            scored['score_filing_speed'] + 
            scored['score_rep_profile'] +
            scored['score_stock_quality']
        )
        
        # 标记为可操作信号
        scored['is_signal'] = (scored['total_score'] >= self.min_score) & buy_mask
        
        return scored

    def get_signals(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """获取最佳交易信号"""
        scored = self.score_trades(df) if 'total_score' not in df.columns else df
        
        signals = scored[scored['is_signal']].copy()
        signals = signals.sort_values('total_score', ascending=False).head(top_n)
        
        return signals

    def _compute_rep_stats(self, df: pd.DataFrame):
        """预计算每位议员的交易统计"""
        self._rep_stats = {}
        for rep, group in df.groupby('representative'):
            buy_count = (group['trade_type'] == 'Purchase').sum()
            sell_count = (group['trade_type'] == 'Sale').sum()
            total = len(group)
            
            # 历史收益 (如果有)
            returns = group.loc[
                (group['trade_type'] == 'Purchase') & (group['return_since_trade'].notna()),
                'return_since_trade'
            ]
            
            win_rate = (returns > 0).mean() if len(returns) >= 3 else 0.5
            avg_return = returns.mean() if len(returns) >= 3 else 0
            
            # 平均交易金额
            avg_amount = group['amount_est'].mean()
            
            self._rep_stats[rep] = {
                'total_trades': total,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'avg_amount': avg_amount,
                'has_return_data': len(returns) >= 3,
            }

    def _score_amount(self, amount: str) -> int:
        """
        金额评分 (0-20分)
        重新校准: 让中等金额也能拿到不错的分
        """
        amount_scores = {
            "$1,001 - $15,000": 4,
            "$15,001 - $50,000": 10,
            "$50,001 - $100,000": 14,
            "$100,001 - $250,000": 17,
            "$250,001 - $500,000": 18,
            "$500,001 - $1,000,000": 19,
            "$1,000,001 - $5,000,000": 20,
            "$5,000,001 - $25,000,000": 20,
            "$25,000,001 - $50,000,000": 20,
        }
        return amount_scores.get(amount, 2)

    def _score_committee(self, representative: str, ticker: str) -> int:
        """
        委员会相关性评分 (0-15分)
        """
        sector = TICKER_SECTOR.get(ticker, None)
        
        # 检查知名议员
        for name_pattern, info in NOTABLE_MEMBERS.items():
            if name_pattern.lower() in representative.lower():
                if sector and sector in info['sectors']:
                    return 15  # 完全匹配
                else:
                    return 8   # 知名议员但行业不匹配
        
        # 非知名议员
        if sector:
            return 6  # 有行业信息但无法确认关联
        return 4  # 无法判断

    def _score_cluster_batch(self, df: pd.DataFrame) -> pd.Series:
        """
        批量计算集群信号评分 (0-25分)
        v2 改进: 窗口扩大到 14 天，并考虑金额加权
        """
        scores = pd.Series(0, index=df.index)
        
        if df.empty:
            return scores
        
        buys = df[df['trade_type'] == 'Purchase'].copy()
        if buys.empty:
            return scores
        
        for ticker, group in buys.groupby('ticker'):
            if len(group) < 2:
                # 只有一笔交易 → 基础分 3
                scores[group.index] = 3
                continue
            
            group_sorted = group.sort_values('transaction_date')
            
            for idx, row in group_sorted.iterrows():
                tx_date = row['transaction_date']
                if pd.isna(tx_date):
                    continue
                
                window_start = tx_date - timedelta(days=self.cluster_window_days)
                window_end = tx_date + timedelta(days=self.cluster_window_days)
                
                window_trades = group_sorted[
                    (group_sorted['transaction_date'] >= window_start) &
                    (group_sorted['transaction_date'] <= window_end)
                ]
                
                unique_buyers = window_trades['representative'].nunique()
                # 金额加权: 如果集群中有大额交易，额外加分
                max_amount = window_trades['amount_est'].max()
                amount_bonus = 2 if max_amount >= 50000 else 0
                
                if unique_buyers >= 5:
                    scores[idx] = 25
                elif unique_buyers >= 4:
                    scores[idx] = min(22 + amount_bonus, 25)
                elif unique_buyers >= 3:
                    scores[idx] = min(18 + amount_bonus, 25)
                elif unique_buyers >= 2:
                    scores[idx] = min(12 + amount_bonus, 25)
                else:
                    scores[idx] = 3  # 单人买入，基础分
        
        return scores

    def _score_filing_speed(self, delay_days) -> int:
        """
        申报速度评分 (0-10分)
        v2: 降低权重到 10 分，减少缺失数据的影响
        """
        if pd.isna(delay_days):
            return 5  # 无数据，中等分

        delay = abs(delay_days)
        if delay <= 5:
            return 10
        elif delay <= 15:
            return 8
        elif delay <= 30:
            return 6
        elif delay <= 45:
            return 4
        else:
            return 2

    def _score_rep_profile(self, representative: str) -> int:
        """
        议员画像评分 (0-15分)
        综合考虑: 知名度 + 交易频率 + 历史胜率
        """
        # 检查知名议员
        for name_pattern, info in NOTABLE_MEMBERS.items():
            if name_pattern.lower() in representative.lower():
                base = info['fame']  # 5-15
                
                # 叠加历史表现
                stats = self._rep_stats.get(representative, {})
                if stats.get('has_return_data'):
                    if stats['win_rate'] >= 0.6:
                        base = min(base + 2, 15)
                    elif stats['win_rate'] < 0.4:
                        base = max(base - 3, 3)
                
                return min(base, 15)
        
        # 非知名议员，根据交易频率评分
        stats = self._rep_stats.get(representative, {})
        total = stats.get('total_trades', 0)
        
        if total >= 50:
            score = 8   # 高频交易者
        elif total >= 20:
            score = 6
        elif total >= 5:
            score = 5
        else:
            score = 3   # 偶尔交易
        
        # 有历史胜率数据就叠加
        if stats.get('has_return_data'):
            if stats['win_rate'] >= 0.65:
                score = min(score + 3, 15)
            elif stats['win_rate'] >= 0.55:
                score = min(score + 1, 15)
        
        return score

    def _score_stock_quality(self, ticker: str) -> int:
        """
        股票质量评分 (0-15分)
        蓝筹大盘股 > 中盘股 > 小盘/冷门股
        
        逻辑: 
        - 蓝筹股的国会交易更有参考价值 (流动性好，信息更公开)
        - 冷门小票可能有噪音
        """
        if ticker in BLUE_CHIPS:
            return 15  # 蓝筹
        elif ticker in MID_CAPS:
            return 10  # 中盘
        elif ticker in TICKER_SECTOR:
            return 7   # 有行业信息的其他股票
        else:
            return 4   # 未知/冷门

    def format_signals(self, signals: pd.DataFrame) -> str:
        """格式化信号输出"""
        if signals.empty:
            return "当前没有达到阈值的信号"
        
        lines = [
            "=" * 90,
            "🎯 国会跟单信号排行",
            "=" * 90,
            f"{'排名':>4s}  {'股票':6s}  {'评分':>4s}  {'交易者':22s}  {'金额':20s}  {'日期':12s}  {'评分明细'}",
            "-" * 90,
        ]
        
        for i, (_, row) in enumerate(signals.iterrows(), 1):
            detail = (
                f"额{row.get('score_amount',0):.0f} "
                f"委{row.get('score_committee',0):.0f} "
                f"群{row.get('score_cluster',0):.0f} "
                f"速{row.get('score_filing_speed',0):.0f} "
                f"人{row.get('score_rep_profile',0):.0f} "
                f"票{row.get('score_stock_quality',0):.0f}"
            )
            tx_date = row['transaction_date'].strftime('%Y-%m-%d') if pd.notna(row['transaction_date']) else 'N/A'
            
            lines.append(
                f"  #{i:<3d} {row['ticker']:6s}  {row['total_score']:>4.0f}  "
                f"{row['representative'][:22]:22s}  {row['amount'][:20]:20s}  "
                f"{tx_date:12s}  {detail}"
            )
        
        lines.append("=" * 90)
        lines.append(f"共 {len(signals)} 个信号 | 最低分数阈值: {self.min_score}")
        return "\n".join(lines)


# ============================================================
# 便捷函数
# ============================================================

def generate_signals(df: pd.DataFrame, min_score: int = 55, top_n: int = 20) -> pd.DataFrame:
    """一键生成交易信号"""
    scorer = SignalScorer(min_score=min_score)
    return scorer.get_signals(df, top_n=top_n)


if __name__ == "__main__":
    from data_fetcher import fetch_congress_trades
    
    df = fetch_congress_trades(days=90)
    scorer = SignalScorer(min_score=45)
    signals = scorer.get_signals(df, top_n=20)
    print(scorer.format_signals(signals))
