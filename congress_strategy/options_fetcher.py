"""
期权异动数据获取模块 - Options Flow Fetcher
=============================================
通过 yfinance 免费获取期权链数据，检测异常成交量

信号逻辑:
- 当某只股票的看涨期权成交量远超持仓量 (Volume >> Open Interest)
  说明有人在大量买入看涨期权 → 可能预期股价上涨
- 重点关注: 近月期权、大额权利金、聪明资金的方向性押注

检测条件:
- Vol/OI > 3 (成交量是持仓量的 3 倍以上)
- 最低总权利金 > $50,000
- 只看看涨期权 (calls)
- 到期日在 7-60 天内 (排除超短期和超长期)
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# 扫描标的列表 — S&P 100 级别的高流动性股票
# 期权异动在高流动性标的上更有参考价值
SCAN_UNIVERSE = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'CRM', 'AVGO',
    'INTC', 'CSCO', 'ORCL', 'ADBE', 'QCOM', 'NFLX', 'NOW', 'PANW', 'INTU', 'MU',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA', 'PYPL', 'COF',
    # Health
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN', 'GILD', 'MRNA',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'MPC', 'EOG',
    # Consumer
    'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'WMT', 'COST',
    # Industrial
    'BA', 'CAT', 'HON', 'GE', 'RTX', 'LMT', 'UPS', 'DE',
    # ETFs (market-wide signals)
    'SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV',
]


class OptionsFetcher:
    """期权异动数据获取器"""

    def __init__(self,
                 min_vol_oi_ratio: float = 3.0,
                 min_premium: float = 50_000,
                 min_expiry_days: int = 7,
                 max_expiry_days: int = 60,
                 cache_hours: int = 2):
        """
        Args:
            min_vol_oi_ratio: 最低成交量/持仓量比率
            min_premium: 最低总权利金 ($)
            min_expiry_days: 最短到期天数
            max_expiry_days: 最长到期天数
            cache_hours: 缓存时间
        """
        self.min_vol_oi_ratio = min_vol_oi_ratio
        self.min_premium = min_premium
        self.min_expiry_days = min_expiry_days
        self.max_expiry_days = max_expiry_days
        self.cache_hours = cache_hours
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def scan_unusual_activity(self, 
                              tickers: Optional[List[str]] = None,
                              use_cache: bool = True) -> pd.DataFrame:
        """
        扫描期权异动
        
        Args:
            tickers: 扫描标的列表，None 使用默认列表
            use_cache: 使用缓存
            
        Returns:
            pd.DataFrame: 异常期权活动列表
        """
        cache_file = DATA_DIR / "unusual_options.csv"

        if use_cache and cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(hours=self.cache_hours):
                df = pd.read_csv(cache_file, parse_dates=['scan_date', 'expiry'])
                print(f"📦 期权异动缓存: {len(df)} 条")
                return df

        if tickers is None:
            tickers = SCAN_UNIVERSE

        print(f"🔄 扫描 {len(tickers)} 只标的的期权异动...")
        all_unusual = []

        for i, ticker in enumerate(tickers):
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(tickers)}...")
            
            try:
                unusual = self._scan_ticker(ticker)
                if unusual:
                    all_unusual.extend(unusual)
            except Exception as e:
                logger.debug(f"{ticker} 扫描失败: {e}")
                continue

        if not all_unusual:
            print("  ⚠️ 未发现异常期权活动 (市场可能已收盘)")
            return pd.DataFrame()

        df = pd.DataFrame(all_unusual)
        df = df.sort_values('total_premium', ascending=False).reset_index(drop=True)

        df.to_csv(cache_file, index=False)
        print(f"📊 期权异动: 发现 {len(df)} 条异常活动 ({df['ticker'].nunique()} 只标的)")
        return df

    def _scan_ticker(self, ticker: str) -> List[Dict]:
        """扫描单只股票的期权异动"""
        results = []
        
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            
            if not expirations:
                return results

            now = datetime.now()

            for exp_str in expirations[:4]:  # 只看最近 4 个到期日
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                    days_to_expiry = (exp_date - now).days

                    if days_to_expiry < self.min_expiry_days or days_to_expiry > self.max_expiry_days:
                        continue

                    chain = t.option_chain(exp_str)
                    
                    # 扫描看涨期权
                    calls_unusual = self._find_unusual(
                        chain.calls, ticker, exp_str, days_to_expiry, 'call'
                    )
                    results.extend(calls_unusual)

                    # 扫描看跌期权 (也有参考价值)
                    puts_unusual = self._find_unusual(
                        chain.puts, ticker, exp_str, days_to_expiry, 'put'
                    )
                    results.extend(puts_unusual)

                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"{ticker} 期权获取失败: {e}")

        return results

    def _find_unusual(self, options_df: pd.DataFrame, ticker: str,
                      expiry: str, days_to_expiry: int,
                      option_type: str) -> List[Dict]:
        """在期权链中查找异常活动"""
        results = []

        if options_df.empty:
            return results

        df = options_df.copy()

        # 过滤条件
        df = df[df['volume'].notna() & (df['volume'] > 0)].copy()
        df = df[df['openInterest'].notna()].copy()

        # 计算 Vol/OI ratio
        df['vol_oi_ratio'] = df['volume'] / df['openInterest'].replace(0, 1)

        # 估算总权利金
        df['total_premium'] = df['volume'] * df['lastPrice'] * 100  # 每份 100 股

        # 筛选异常
        unusual = df[
            (df['vol_oi_ratio'] >= self.min_vol_oi_ratio) &
            (df['total_premium'] >= self.min_premium) &
            (df['openInterest'] > 0)  # 排除新上市期权 (OI=0)
        ].copy()

        for _, row in unusual.iterrows():
            results.append({
                'ticker': ticker,
                'option_type': option_type,
                'strike': row['strike'],
                'expiry': expiry,
                'days_to_expiry': days_to_expiry,
                'volume': int(row['volume']),
                'open_interest': int(row['openInterest']),
                'vol_oi_ratio': round(row['vol_oi_ratio'], 2),
                'last_price': round(row['lastPrice'], 2),
                'total_premium': round(row['total_premium'], 0),
                'implied_volatility': round(row.get('impliedVolatility', 0), 4),
                'in_the_money': bool(row.get('inTheMoney', False)),
                'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            })

        return results


def scan_options_flow(tickers: Optional[List[str]] = None, 
                      use_cache: bool = True,
                      extra_tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    一键扫描期权异动
    
    Args:
        tickers: 主扫描标的列表，None 使用默认列表
        use_cache: 使用缓存
        extra_tickers: 额外补扫的 ticker 列表（来自国会/内部人数据）
    """
    fetcher = OptionsFetcher()
    result = fetcher.scan_unusual_activity(tickers, use_cache)
    
    # 动态补扫：对不在默认宇宙中的 ticker 额外扫描
    if extra_tickers:
        existing_universe = set(tickers or SCAN_UNIVERSE)
        new_tickers = [t for t in extra_tickers if t not in existing_universe]
        if new_tickers:
            print(f"🔄 动态补扫 {len(new_tickers)} 只额外标的 (来自国会/内部人)...")
            extra_result = fetcher.scan_unusual_activity(
                tickers=new_tickers, use_cache=False
            )
            if not extra_result.empty:
                print(f"  ✅ 补扫发现 {len(extra_result)} 条额外异动")
                if not result.empty:
                    result = pd.concat([result, extra_result], ignore_index=True)
                    result = result.sort_values('total_premium', ascending=False).reset_index(drop=True)
                else:
                    result = extra_result
    
    return result


def get_extra_tickers_from_signals(congress_df=None, insider_df=None) -> List[str]:
    """
    从国会和内部人数据中提取额外需要扫描期权的 ticker
    
    Returns:
        List[str]: 不在默认扫描宇宙中的 ticker 列表
    """
    extra = set()
    existing = set(SCAN_UNIVERSE)
    
    if congress_df is not None and not congress_df.empty:
        buys = congress_df[congress_df['trade_type'].str.contains('Purchase', case=False, na=False)]
        congress_tickers = set(buys['ticker'].unique())
        extra.update(congress_tickers - existing)
    
    if insider_df is not None and not insider_df.empty:
        insider_tickers = set(insider_df['ticker'].unique())
        extra.update(insider_tickers - existing)
    
    return sorted(extra)


if __name__ == "__main__":
    df = scan_options_flow(use_cache=False)
    if not df.empty:
        print(f"\nTop 10 异常活动:")
        print(df[['ticker', 'option_type', 'strike', 'expiry', 'volume', 'open_interest', 
                   'vol_oi_ratio', 'total_premium']].head(10).to_string())
