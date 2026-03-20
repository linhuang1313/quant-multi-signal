"""
多信号共振引擎 - Multi-Signal Resonance Engine v2.1
====================================================
整合三大信号源，当多个信号指向同一只股票时产生强共振信号

v2.1 改进:
- 国会信号窗口从14天扩大到45天，增加时效衰减（越新越强）
- 期权扫描动态扩展：除固定宇宙外，自动补扫国会+内部人出现的ticker
- 阈值策略优化：共振信号和单信号分别设置不同阈值
- 国会信号过滤放宽：接受 Sale (Partial) 以外的所有交易

信号源:
1. 国会议员交易 (Congress Trades) — 信息优势
2. 内部人买入 (Insider Purchases) — 知情人信心
3. 期权异动 (Unusual Options Activity) — 聪明资金方向

共振逻辑:
- 单信号 (1个来源) → 弱信号，高阈值过滤
- 双信号 (2个来源) → 中强信号，较低阈值
- 三重信号 (3个来源) → 最强信号，优先交易

评分 = 基础分(时效衰减) + 共振加成 + 各信号源内部评分
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# 共振加成分数
RESONANCE_BONUS = {
    1: 0,    # 单信号，无加成
    2: 25,   # 双信号共振
    3: 50,   # 三重共振
}

# 不同共振等级的最低评分（核心改动：共振信号阈值更低，更容易被捕捉到）
RESONANCE_MIN_SCORE = {
    1: 40,   # 单信号需要 40 分才有价值
    2: 30,   # 双重共振只需 30 分（共振本身就是强信号）
    3: 20,   # 三重共振 20 分即可（极其罕见，出现即有意义）
}

# 信号源权重
SOURCE_WEIGHTS = {
    'congress': 1.0,       # 国会交易
    'insider': 1.2,        # 内部人买入 (权重略高，因为是最了解公司的人)
    'options': 0.8,        # 期权异动 (权重略低，噪音较多)
}

# 时效衰减配置（天数 → 衰减因子）
# 越新的信号越强
RECENCY_DECAY = [
    (7,  1.0),    # 7天内: 满分
    (14, 0.85),   # 8-14天: 85%
    (21, 0.70),   # 15-21天: 70%
    (30, 0.55),   # 22-30天: 55%
    (45, 0.40),   # 31-45天: 40%
]


def _recency_factor(days_ago: float) -> float:
    """计算时效衰减因子"""
    for max_days, factor in RECENCY_DECAY:
        if days_ago <= max_days:
            return factor
    return 0.25  # 超过45天: 25%


class MultiSignalEngine:
    """多信号共振引擎 v2.1"""

    def __init__(self,
                 congress_lookback_days: int = 45,
                 min_score: int = 30):
        """
        Args:
            congress_lookback_days: 国会信号回看窗口（天），默认45天
            min_score: 全局最低评分（实际按共振等级动态调整）
        """
        self.congress_lookback_days = congress_lookback_days
        self.min_score = min_score

    def generate_signals(self,
                         congress_df: pd.DataFrame,
                         insider_df: pd.DataFrame,
                         options_df: pd.DataFrame,
                         top_n: int = 20) -> pd.DataFrame:
        """
        生成多信号共振交易信号
        
        Args:
            congress_df: 国会交易数据
            insider_df: 内部人买入数据
            options_df: 期权异动数据
            top_n: 返回前 N 个信号
            
        Returns:
            pd.DataFrame: 综合评分后的交易信号
        """
        print("\n" + "=" * 65)
        print("🔮 多信号共振引擎 v2.1")
        print("=" * 65)

        # Step 1: 提取各信号源的看涨信号
        congress_signals = self._extract_congress_signals(congress_df)
        insider_signals = self._extract_insider_signals(insider_df)
        options_signals = self._extract_options_signals(options_df)

        print(f"\n📡 信号源状态:")
        print(f"  🏛️ 国会交易:  {len(congress_signals)} 个看涨信号 (回看 {self.congress_lookback_days} 天)")
        print(f"  👔 内部人买入: {len(insider_signals)} 个看涨信号")
        print(f"  📊 期权异动:  {len(options_signals)} 个看涨信号")

        # Step 2: 合并所有 ticker，检测共振
        all_tickers = set()
        if not congress_signals.empty:
            all_tickers.update(congress_signals['ticker'].unique())
        if not insider_signals.empty:
            all_tickers.update(insider_signals['ticker'].unique())
        if not options_signals.empty:
            all_tickers.update(options_signals['ticker'].unique())

        if not all_tickers:
            print("\n⚠️ 没有找到任何信号")
            return pd.DataFrame()

        print(f"\n🎯 涉及 {len(all_tickers)} 只独立标的")

        # Step 2.5: 预先检测重叠 ticker（方便调试）
        congress_tickers = set(congress_signals['ticker'].unique()) if not congress_signals.empty else set()
        insider_tickers = set(insider_signals['ticker'].unique()) if not insider_signals.empty else set()
        options_tickers = set(options_signals['ticker'].unique()) if not options_signals.empty else set()
        
        overlap_ci = congress_tickers & insider_tickers
        overlap_co = congress_tickers & options_tickers
        overlap_io = insider_tickers & options_tickers
        triple = congress_tickers & insider_tickers & options_tickers
        
        print(f"\n📊 信号重叠分析:")
        print(f"  国会+内部人: {sorted(overlap_ci) if overlap_ci else '无'}")
        print(f"  国会+期权:  {sorted(overlap_co) if overlap_co else '无'}")
        print(f"  内部人+期权: {sorted(overlap_io) if overlap_io else '无'}")
        print(f"  三重重叠:   {sorted(triple) if triple else '无'}")

        # Step 3: 逐 ticker 计算共振评分
        scored_signals = []
        for ticker in all_tickers:
            signal = self._score_ticker(
                ticker, congress_signals, insider_signals, options_signals
            )
            if signal:
                # 按共振等级使用不同的阈值
                level = signal['resonance_level']
                threshold = RESONANCE_MIN_SCORE.get(level, self.min_score)
                if signal['total_score'] >= threshold:
                    scored_signals.append(signal)

        if not scored_signals:
            print("\n⚠️ 没有达到阈值的信号")
            return pd.DataFrame()

        # Step 4: 排序输出 — 先按共振等级降序，再按总分降序
        result = pd.DataFrame(scored_signals)
        result = result.sort_values(
            ['resonance_level', 'total_score'], 
            ascending=[False, False]
        ).head(top_n)
        result = result.reset_index(drop=True)

        # 统计
        resonance_counts = result['resonance_level'].value_counts().to_dict()
        print(f"\n✨ 信号共振统计:")
        print(f"  三重共振 🔴🔴🔴: {resonance_counts.get(3, 0)} 个")
        print(f"  双重共振 🔴🔴:   {resonance_counts.get(2, 0)} 个")
        print(f"  单信号   🔴:     {resonance_counts.get(1, 0)} 个")
        print(f"  总计: {len(result)} 个信号")

        return result

    def _extract_congress_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取国会买入信号 (v2.1: 窗口扩大 + 时效衰减)"""
        if df is None or df.empty:
            return pd.DataFrame()

        cutoff = datetime.now() - timedelta(days=self.congress_lookback_days)
        
        # 放宽过滤: 接受 "Purchase" 交易
        # 注意: QuiverQuant 数据中 trade_type 可能是 "Purchase", "Sale", 等
        buys = df[
            (df['trade_type'].str.contains('Purchase', case=False, na=False)) &
            (df['transaction_date'] >= cutoff)
        ].copy()

        if buys.empty:
            return pd.DataFrame()

        # 计算国会内部评分（含时效衰减）
        now = datetime.now()
        scores = []
        for _, row in buys.iterrows():
            score = 0
            
            # 金额评分
            amt = row.get('amount_est', 8000)
            if amt >= 100000:
                score += 30
            elif amt >= 50000:
                score += 20
            elif amt >= 15000:
                score += 12
            else:
                score += 5

            # 知名度加分
            rep = str(row.get('representative', ''))
            if any(n in rep for n in ['Pelosi', 'McConnell', 'Loeffler', 'Tuberville', 'McCaul']):
                score += 15
            else:
                score += 5

            # 多笔买入同一ticker的加成
            same_ticker_count = len(buys[buys['ticker'] == row['ticker']])
            if same_ticker_count >= 3:
                score += 10  # 3+个议员买同一只
            elif same_ticker_count >= 2:
                score += 5   # 2个议员买同一只
            
            # 时效衰减
            tx_date = row.get('transaction_date')
            if pd.notna(tx_date):
                days_ago = (now - pd.Timestamp(tx_date)).total_seconds() / 86400
                score = score * _recency_factor(days_ago)
            
            scores.append(round(score))

        buys = buys.copy()
        buys['source_score'] = scores
        buys['signal_source'] = 'congress'

        # 按 ticker 聚合 — 同一 ticker 多笔交易取最高分，但记录交易数
        agg = buys.groupby('ticker').agg({
            'source_score': 'max',
            'signal_source': 'first',
            'transaction_date': 'max',
            'representative': lambda x: ', '.join(x.unique()[:3]),  # 最多显示3个议员名
            'amount': 'first',
        }).reset_index()

        # 添加交易笔数
        trade_counts = buys.groupby('ticker').size().reset_index(name='congress_trade_count')
        agg = agg.merge(trade_counts, on='ticker', how='left')
        
        agg = agg.rename(columns={'transaction_date': 'signal_date'})
        return agg

    def _extract_insider_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取内部人买入信号"""
        if df is None or df.empty:
            return pd.DataFrame()

        signals = df.copy()
        
        # 计算内部人评分
        scores = []
        for _, row in signals.iterrows():
            score = 0
            
            # 金额
            val = row.get('value', 0)
            if val >= 1_000_000:
                score += 35
            elif val >= 500_000:
                score += 28
            elif val >= 100_000:
                score += 20
            elif val >= 50_000:
                score += 12
            else:
                score += 5

            # 集群加成
            count = row.get('insider_count', 1)
            if count >= 4:
                score += 25
            elif count >= 3:
                score += 18
            elif count >= 2:
                score += 10

            # 职位加成
            title = str(row.get('title', '')).upper()
            if any(t in title for t in ['CEO', 'COB', 'PRES']):
                score += 10
            elif any(t in title for t in ['CFO', 'COO', 'CTO']):
                score += 8
            elif 'DIR' in title:
                score += 5
            elif '10%' in title:
                score += 7

            scores.append(score)

        signals = signals.copy()
        signals['source_score'] = scores
        signals['signal_source'] = 'insider'

        return signals[['ticker', 'signal_source', 'source_score', 'trade_date',
                         'insider_name', 'value']].rename(
            columns={'trade_date': 'signal_date'}
        )

    def _extract_options_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取期权异动看涨信号"""
        if df is None or df.empty:
            return pd.DataFrame()

        # 只看看涨期权的异动 (calls)
        calls = df[df['option_type'] == 'call'].copy()
        if calls.empty:
            return pd.DataFrame()

        # 按 ticker 聚合 (同一只股票可能有多个异常期权)
        agg = calls.groupby('ticker').agg({
            'total_premium': 'sum',
            'vol_oi_ratio': 'max',
            'volume': 'sum',
            'scan_date': 'first',
        }).reset_index()

        # 评分
        scores = []
        for _, row in agg.iterrows():
            score = 0
            
            # 总权利金
            prem = row['total_premium']
            if prem >= 5_000_000:
                score += 30
            elif prem >= 1_000_000:
                score += 22
            elif prem >= 500_000:
                score += 15
            elif prem >= 100_000:
                score += 8
            else:
                score += 3

            # Vol/OI 极端程度
            ratio = row['vol_oi_ratio']
            if ratio >= 10:
                score += 15
            elif ratio >= 5:
                score += 10
            elif ratio >= 3:
                score += 5

            scores.append(score)

        agg['source_score'] = scores
        agg['signal_source'] = 'options'
        agg['signal_date'] = pd.to_datetime(agg['scan_date'], errors='coerce')

        return agg[['ticker', 'signal_source', 'source_score', 'signal_date', 
                     'total_premium', 'vol_oi_ratio']]

    def _score_ticker(self, ticker: str,
                      congress: pd.DataFrame,
                      insider: pd.DataFrame,
                      options: pd.DataFrame) -> Optional[Dict]:
        """计算单只 ticker 的综合共振评分"""
        
        sources_present = []
        details = {}
        total_source_score = 0

        # 检查国会信号
        if not congress.empty:
            c = congress[congress['ticker'] == ticker]
            if not c.empty:
                sources_present.append('congress')
                best = c.sort_values('source_score', ascending=False).iloc[0]
                score = best['source_score'] * SOURCE_WEIGHTS['congress']
                total_source_score += score
                details['congress_score'] = round(score)
                details['congress_rep'] = best.get('representative', '')
                details['congress_amount'] = best.get('amount', '')
                details['congress_trades'] = best.get('congress_trade_count', 1)

        # 检查内部人信号
        if not insider.empty:
            ins = insider[insider['ticker'] == ticker]
            if not ins.empty:
                sources_present.append('insider')
                best = ins.sort_values('source_score', ascending=False).iloc[0]
                score = best['source_score'] * SOURCE_WEIGHTS['insider']
                total_source_score += score
                details['insider_score'] = round(score)
                details['insider_name'] = best.get('insider_name', '')
                details['insider_value'] = best.get('value', 0)

        # 检查期权信号
        if not options.empty:
            opt = options[options['ticker'] == ticker]
            if not opt.empty:
                sources_present.append('options')
                best = opt.sort_values('source_score', ascending=False).iloc[0]
                score = best['source_score'] * SOURCE_WEIGHTS['options']
                total_source_score += score
                details['options_score'] = round(score)
                details['options_premium'] = best.get('total_premium', 0)
                details['options_vol_oi'] = best.get('vol_oi_ratio', 0)

        if not sources_present:
            return None

        # 共振等级和加成
        resonance_level = len(sources_present)
        resonance_bonus = RESONANCE_BONUS.get(resonance_level, 0)

        total_score = total_source_score + resonance_bonus

        return {
            'ticker': ticker,
            'total_score': round(total_score),
            'resonance_level': resonance_level,
            'resonance_bonus': resonance_bonus,
            'source_score': round(total_source_score),
            'sources': '+'.join(sources_present),
            'num_sources': resonance_level,
            **details,
        }

    @staticmethod
    def format_signals(signals: pd.DataFrame) -> str:
        """格式化输出信号"""
        if signals.empty:
            return "无信号"

        resonance_icons = {1: '🔴', 2: '🔴🔴', 3: '🔴🔴🔴'}

        lines = [
            "",
            "=" * 90,
            "🔮 多信号共振交易信号 (v2.1)",
            "=" * 90,
            f"{'#':>3s}  {'共振':8s}  {'股票':6s}  {'总分':>5s}  {'信号源':25s}  {'明细'}",
            "-" * 90,
        ]

        for i, (_, row) in enumerate(signals.iterrows(), 1):
            icon = resonance_icons.get(row['resonance_level'], '⚪')
            
            detail_parts = []
            if 'congress_score' in row and pd.notna(row.get('congress_score')):
                rep = str(row.get('congress_rep', '')).replace('nan', '')[:20]
                trades = row.get('congress_trades', 1)
                trades_str = f",{int(trades)}笔" if pd.notna(trades) and trades > 1 else ""
                detail_parts.append(f"国会{row['congress_score']:.0f}({rep}{trades_str})")
            if 'insider_score' in row and pd.notna(row.get('insider_score')):
                val = row.get('insider_value', 0)
                val = val if pd.notna(val) else 0
                name = str(row.get('insider_name', '')).replace('nan', '')
                name_str = f"{name}" if name else ""
                detail_parts.append(f"内部人{row['insider_score']:.0f}({name_str}${val:,.0f})")
            if 'options_score' in row and pd.notna(row.get('options_score')):
                prem = row.get('options_premium', 0)
                prem = prem if pd.notna(prem) else 0
                vol_oi = row.get('options_vol_oi', 0)
                vol_oi = vol_oi if pd.notna(vol_oi) else 0
                detail_parts.append(f"期权{row['options_score']:.0f}(${prem:,.0f},V/OI:{vol_oi:.1f})")
            
            detail = ' | '.join(detail_parts)

            lines.append(
                f"  {i:>2d}  {icon:8s}  {row['ticker']:6s}  {row['total_score']:>5.0f}  "
                f"{row['sources']:25s}  {detail}"
            )

        lines.extend([
            "-" * 90,
            f"共 {len(signals)} 个信号 | "
            f"三重共振: {(signals['resonance_level']==3).sum()} | "
            f"双重共振: {(signals['resonance_level']==2).sum()} | "
            f"单信号: {(signals['resonance_level']==1).sum()}",
            "=" * 90,
        ])

        return "\n".join(lines)


class ShortSignalEngine:
    """
    做空信号引擎 — 极高确定性标准
    
    只有满足以下条件才产生做空信号:
    1. 必须至少有2个信号源同时看空（双重共振以上）
    2. 每个信号源内部必须有高分（过滤噪音卖出）
    3. 做空总分必须 >= 50（比做多的30更严格）
    
    做空信号源:
    - 国会大额卖出（多位议员集中卖出同一标的）
    - 内部人大额卖出（CEO/CFO 级别，金额 > $500k）
    - 看跌期权异动（Vol/OI > 5，权利金 > $200k）
    """

    # 做空共振加成
    SHORT_RESONANCE_BONUS = {
        1: 0,
        2: 30,   # 双重共振
        3: 60,   # 三重共振
    }

    # 做空最低阈值 — 比做多更严格
    SHORT_MIN_SCORE = {
        1: 999,  # 单信号永远不做空（太多噪音）
        2: 50,   # 双重共振需要 50 分
        3: 40,   # 三重共振 40 分
    }

    def __init__(self, congress_lookback_days: int = 45):
        self.congress_lookback_days = congress_lookback_days

    def generate_short_signals(self,
                                congress_df: pd.DataFrame,
                                insider_sales_df: pd.DataFrame,
                                options_df: pd.DataFrame,
                                top_n: int = 10) -> pd.DataFrame:
        """生成做空信号 — 仅输出双重/三重共振"""
        print("\n" + "=" * 65)
        print("🔻 做空信号引擎 (高确定性模式)")
        print("=" * 65)

        congress_shorts = self._extract_congress_sell_signals(congress_df)
        insider_shorts = self._extract_insider_sell_signals(insider_sales_df)
        options_shorts = self._extract_put_signals(options_df)

        print(f"\n📡 做空信号源状态:")
        print(f"  🏛️ 国会卖出:  {len(congress_shorts)} 个看空信号")
        print(f"  👔 内部人卖出: {len(insider_shorts)} 个看空信号")
        print(f"  📊 看跌期权:  {len(options_shorts)} 个看空信号")

        # 合并 ticker
        all_tickers = set()
        for df in [congress_shorts, insider_shorts, options_shorts]:
            if not df.empty:
                all_tickers.update(df['ticker'].unique())

        if not all_tickers:
            print("\n⚠️ 没有找到做空信号")
            return pd.DataFrame()

        # 检测重叠
        c_tickers = set(congress_shorts['ticker'].unique()) if not congress_shorts.empty else set()
        i_tickers = set(insider_shorts['ticker'].unique()) if not insider_shorts.empty else set()
        o_tickers = set(options_shorts['ticker'].unique()) if not options_shorts.empty else set()
        
        print(f"\n📊 做空重叠分析:")
        print(f"  国会卖出+内部人卖出: {sorted(c_tickers & i_tickers) or '无'}")
        print(f"  国会卖出+看跌期权:  {sorted(c_tickers & o_tickers) or '无'}")
        print(f"  内部人卖出+看跌期权: {sorted(i_tickers & o_tickers) or '无'}")
        print(f"  三重做空重叠:       {sorted(c_tickers & i_tickers & o_tickers) or '无'}")

        # 评分
        scored = []
        for ticker in all_tickers:
            signal = self._score_short_ticker(
                ticker, congress_shorts, insider_shorts, options_shorts
            )
            if signal:
                level = signal['resonance_level']
                threshold = self.SHORT_MIN_SCORE.get(level, 999)
                if signal['total_score'] >= threshold:
                    scored.append(signal)

        if not scored:
            print("\n⚠️ 没有达到做空确定性阈值的信号 (需要双重共振 >= 50分)")
            return pd.DataFrame()

        result = pd.DataFrame(scored)
        result = result.sort_values(
            ['resonance_level', 'total_score'],
            ascending=[False, False]
        ).head(top_n).reset_index(drop=True)

        r_counts = result['resonance_level'].value_counts().to_dict()
        print(f"\n🔻 做空信号统计:")
        print(f"  三重做空 🔵🔵🔵: {r_counts.get(3, 0)} 个")
        print(f"  双重做空 🔵🔵:   {r_counts.get(2, 0)} 个")
        print(f"  总计: {len(result)} 个做空信号")

        return result

    def _extract_congress_sell_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取国会卖出信号 — 只看集中卖出和大额卖出"""
        if df is None or df.empty:
            return pd.DataFrame()

        cutoff = datetime.now() - timedelta(days=self.congress_lookback_days)
        sells = df[
            (df['trade_type'] == 'Sale') &
            (df['transaction_date'] >= cutoff)
        ].copy()

        if sells.empty:
            return pd.DataFrame()

        # 按 ticker 聚合
        agg = sells.groupby('ticker').agg({
            'representative': lambda x: list(x.unique()),
            'amount_est': 'sum',
            'transaction_date': 'max',
        }).reset_index()

        agg['num_sellers'] = agg['representative'].apply(len)
        
        # 严格过滤: 至少2位议员卖出 OR 单人大额卖出(>$100k)
        agg = agg[(agg['num_sellers'] >= 2) | (agg['amount_est'] >= 100000)].copy()

        if agg.empty:
            return pd.DataFrame()

        scores = []
        for _, row in agg.iterrows():
            score = 0
            # 多人卖出加成
            if row['num_sellers'] >= 4:
                score += 25
            elif row['num_sellers'] >= 3:
                score += 18
            elif row['num_sellers'] >= 2:
                score += 12
            
            # 金额加成
            if row['amount_est'] >= 500000:
                score += 20
            elif row['amount_est'] >= 100000:
                score += 12
            elif row['amount_est'] >= 50000:
                score += 5

            scores.append(score)

        agg['source_score'] = scores
        agg['signal_source'] = 'congress_sell'
        agg['representative'] = agg['representative'].apply(lambda x: ', '.join(x[:3]))
        agg = agg.rename(columns={'transaction_date': 'signal_date', 'amount_est': 'total_amount'})

        return agg[['ticker', 'signal_source', 'source_score', 'signal_date',
                     'representative', 'total_amount', 'num_sellers']]

    def _extract_insider_sell_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取内部人卖出信号 — 只看大额+高管卖出"""
        if df is None or df.empty:
            return pd.DataFrame()

        signals = df.copy()
        scores = []
        for _, row in signals.iterrows():
            score = 0
            val = row.get('value', 0)
            # 金额门槛极高 — 过滤掉常规行权套现
            if val >= 5_000_000:
                score += 30
            elif val >= 1_000_000:
                score += 22
            elif val >= 500_000:
                score += 15
            else:
                score += 0  # < $500k 的卖出不计分（太多噪音）

            # CEO/CFO 级别卖出更有意义
            title = str(row.get('title', '')).upper()
            if any(t in title for t in ['CEO', 'COB', 'PRES']):
                score += 12
            elif any(t in title for t in ['CFO', 'COO', 'CTO']):
                score += 10
            elif 'DIR' in title:
                score += 5

            # 持股比例大幅下降
            own_change = row.get('ownership_change', 0)
            if pd.notna(own_change) and own_change < -30:
                score += 10  # 减持超30%
            elif pd.notna(own_change) and own_change < -15:
                score += 5

            scores.append(score)

        signals = signals.copy()
        signals['source_score'] = scores
        signals['signal_source'] = 'insider_sell'

        # 只保留有分数的（过滤掉小额噪音卖出）
        signals = signals[signals['source_score'] > 0].copy()

        if signals.empty:
            return pd.DataFrame()

        return signals[['ticker', 'signal_source', 'source_score', 'trade_date',
                         'insider_name', 'value', 'title']].rename(
            columns={'trade_date': 'signal_date'}
        )

    def _extract_put_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取看跌期权异动 — 高门槛"""
        if df is None or df.empty:
            return pd.DataFrame()

        puts = df[df['option_type'] == 'put'].copy()
        if puts.empty:
            return pd.DataFrame()

        agg = puts.groupby('ticker').agg({
            'total_premium': 'sum',
            'vol_oi_ratio': 'max',
            'volume': 'sum',
            'scan_date': 'first',
        }).reset_index()

        # 更高门槛: 权利金 > $200k 或 Vol/OI > 5
        agg = agg[(agg['total_premium'] >= 200000) | (agg['vol_oi_ratio'] >= 5)].copy()

        if agg.empty:
            return pd.DataFrame()

        scores = []
        for _, row in agg.iterrows():
            score = 0
            prem = row['total_premium']
            if prem >= 5_000_000:
                score += 30
            elif prem >= 1_000_000:
                score += 22
            elif prem >= 500_000:
                score += 15
            elif prem >= 200_000:
                score += 8

            ratio = row['vol_oi_ratio']
            if ratio >= 10:
                score += 15
            elif ratio >= 5:
                score += 10

            scores.append(score)

        agg['source_score'] = scores
        agg['signal_source'] = 'put_options'
        agg['signal_date'] = pd.to_datetime(agg['scan_date'], errors='coerce')

        return agg[['ticker', 'signal_source', 'source_score', 'signal_date',
                     'total_premium', 'vol_oi_ratio']]

    def _score_short_ticker(self, ticker, congress, insider, options):
        """评分单只 ticker 的做空信号"""
        sources = []
        details = {}
        total_source_score = 0

        if not congress.empty:
            c = congress[congress['ticker'] == ticker]
            if not c.empty:
                sources.append('congress_sell')
                best = c.sort_values('source_score', ascending=False).iloc[0]
                score = best['source_score'] * SOURCE_WEIGHTS['congress']
                total_source_score += score
                details['congress_score'] = round(score)
                details['congress_rep'] = best.get('representative', '')
                details['congress_sellers'] = best.get('num_sellers', 0)

        if not insider.empty:
            ins = insider[insider['ticker'] == ticker]
            if not ins.empty:
                sources.append('insider_sell')
                best = ins.sort_values('source_score', ascending=False).iloc[0]
                score = best['source_score'] * SOURCE_WEIGHTS['insider']
                total_source_score += score
                details['insider_score'] = round(score)
                details['insider_name'] = best.get('insider_name', '')
                details['insider_value'] = best.get('value', 0)

        if not options.empty:
            opt = options[options['ticker'] == ticker]
            if not opt.empty:
                sources.append('put_options')
                best = opt.sort_values('source_score', ascending=False).iloc[0]
                score = best['source_score'] * SOURCE_WEIGHTS['options']
                total_source_score += score
                details['options_score'] = round(score)
                details['options_premium'] = best.get('total_premium', 0)
                details['options_vol_oi'] = best.get('vol_oi_ratio', 0)

        if not sources:
            return None

        resonance_level = len(sources)
        resonance_bonus = self.SHORT_RESONANCE_BONUS.get(resonance_level, 0)
        total_score = total_source_score + resonance_bonus

        return {
            'ticker': ticker,
            'direction': 'SHORT',
            'total_score': round(total_score),
            'resonance_level': resonance_level,
            'resonance_bonus': resonance_bonus,
            'source_score': round(total_source_score),
            'sources': '+'.join(sources),
            'num_sources': resonance_level,
            **details,
        }

    @staticmethod
    def format_short_signals(signals: pd.DataFrame) -> str:
        if signals.empty:
            return "无做空信号"

        icons = {2: '🔵🔵', 3: '🔵🔵🔵'}
        lines = ["", "=" * 90, "🔻 做空共振信号 (高确定性)", "=" * 90,
                  f"{'#':>3s}  {'共振':8s}  {'股票':6s}  {'总分':>5s}  {'信号源':30s}  {'明细'}",
                  "-" * 90]

        for i, (_, row) in enumerate(signals.iterrows(), 1):
            icon = icons.get(row['resonance_level'], '🔵')
            detail_parts = []
            if pd.notna(row.get('congress_score')):
                sellers = row.get('congress_sellers', 0)
                detail_parts.append(f"国会卖出{row['congress_score']:.0f}({int(sellers)}人)")
            if pd.notna(row.get('insider_score')):
                val = row.get('insider_value', 0)
                val = val if pd.notna(val) else 0
                detail_parts.append(f"内部人卖{row['insider_score']:.0f}(${val:,.0f})")
            if pd.notna(row.get('options_score')):
                prem = row.get('options_premium', 0)
                prem = prem if pd.notna(prem) else 0
                detail_parts.append(f"看跌期权{row['options_score']:.0f}(${prem:,.0f})")
            detail = ' | '.join(detail_parts)

            lines.append(f"  {i:>2d}  {icon:8s}  {row['ticker']:6s}  {row['total_score']:>5.0f}  "
                         f"{row['sources']:30s}  {detail}")

        lines.extend(["-" * 90,
                      f"共 {len(signals)} 个做空信号 (仅显示双重/三重共振)",
                      "=" * 90])
        return "\n".join(lines)


if __name__ == "__main__":
    from data_fetcher import fetch_congress_trades
    from insider_fetcher import fetch_insider_trades, InsiderFetcher
    from options_fetcher import scan_options_flow, get_extra_tickers_from_signals

    congress = fetch_congress_trades(days=60)
    insider = fetch_insider_trades(use_cache=False)

    # 动态补扫
    extra = get_extra_tickers_from_signals(congress, insider)
    print(f"\n📡 动态补扫额外标的: {len(extra)} 只")
    options = scan_options_flow(use_cache=False, extra_tickers=extra)

    # === 做多信号 ===
    engine = MultiSignalEngine(congress_lookback_days=45, min_score=30)
    signals = engine.generate_signals(congress, insider, options, top_n=20)
    print(MultiSignalEngine.format_signals(signals))

    # === 做空信号 ===
    insider_sales = InsiderFetcher().fetch_sales(use_cache=False)
    short_engine = ShortSignalEngine(congress_lookback_days=45)
    short_signals = short_engine.generate_short_signals(congress, insider_sales, options, top_n=10)
    print(ShortSignalEngine.format_short_signals(short_signals))
