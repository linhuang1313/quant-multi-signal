"""
内部人交易数据获取模块 - Insider Trading Fetcher
=================================================
从 OpenInsider.com 获取 SEC Form 4 内部人买入数据（免费）

数据源:
1. OpenInsider 最新买入 (Latest Insider Purchases)
2. OpenInsider 集群买入 (Cluster Buys) — 最强信号
3. OpenInsider 大额买入筛选器 (Screener)

信号逻辑:
- CEO/CFO/Director 用自己的钱买入自家股票 = 对公司前景有信心
- 集群买入（多位高管同时买入）= 非常强的信号
- 金额越大 = 信号越强
"""

import re
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from io import StringIO

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

DATA_DIR = Path(__file__).parent / "data"


class InsiderFetcher:
    """SEC Form 4 内部人交易数据获取器"""

    def __init__(self, cache_hours: int = 4):
        self.cache_hours = cache_hours
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_all(self, use_cache: bool = True) -> pd.DataFrame:
        """获取所有内部人买入数据"""
        cache_file = DATA_DIR / "insider_trades.csv"

        if use_cache and cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(hours=self.cache_hours):
                df = pd.read_csv(cache_file, parse_dates=['filing_date', 'trade_date'])
                print(f"📦 内部人数据缓存: {len(df)} 笔")
                return df

        print("🔄 获取内部人交易数据...")
        dfs = []

        # 1. 集群买入 (最强信号)
        df_cluster = self._fetch_cluster_buys()
        if df_cluster is not None and len(df_cluster) > 0:
            df_cluster['signal_type'] = 'cluster_buy'
            dfs.append(df_cluster)
            print(f"  ✅ 集群买入: {len(df_cluster)} 笔")

        # 2. 最新买入
        df_purchases = self._fetch_latest_purchases()
        if df_purchases is not None and len(df_purchases) > 0:
            df_purchases['signal_type'] = 'insider_purchase'
            dfs.append(df_purchases)
            print(f"  ✅ 最新买入: {len(df_purchases)} 笔")

        if not dfs:
            print("❌ 无法获取内部人数据")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['ticker', 'insider_name', 'trade_date'], keep='first')
        df = df.sort_values('filing_date', ascending=False).reset_index(drop=True)

        df.to_csv(cache_file, index=False)
        print(f"📊 内部人数据: 共 {len(df)} 笔有效交易")
        return df

    def _fetch_cluster_buys(self) -> Optional[pd.DataFrame]:
        """获取集群买入数据"""
        try:
            resp = requests.get(
                "http://openinsider.com/latest-cluster-buys",
                headers=HEADERS, timeout=15
            )
            resp.raise_for_status()
            return self._parse_openinsider_table(resp.text, is_cluster=True)
        except Exception as e:
            logger.error(f"集群买入获取失败: {e}")
            return None

    def _fetch_latest_purchases(self) -> Optional[pd.DataFrame]:
        """获取最新内部人买入"""
        try:
            resp = requests.get(
                "http://openinsider.com/insider-purchases",
                headers=HEADERS, timeout=15
            )
            resp.raise_for_status()
            return self._parse_openinsider_table(resp.text, is_cluster=False)
        except Exception as e:
            logger.error(f"最新买入获取失败: {e}")
            return None

    def _parse_openinsider_table(self, html: str, is_cluster: bool = False) -> Optional[pd.DataFrame]:
        """解析 OpenInsider HTML 表格"""
        try:
            dfs = pd.read_html(StringIO(html))
            
            # 找到主数据表 (行数最多且列数 > 10)
            main_df = None
            for tdf in dfs:
                if tdf.shape[0] >= 5 and tdf.shape[1] >= 12:
                    main_df = tdf
                    break

            if main_df is None:
                return None

            # 标准化列名 (去除 \xa0)
            main_df.columns = [c.replace('\xa0', '_') for c in main_df.columns]

            result = pd.DataFrame()
            result['ticker'] = main_df['Ticker'].str.strip()
            result['company_name'] = main_df.get('Company_Name', '')
            result['filing_date'] = pd.to_datetime(
                main_df['Filing_Date'].str.strip(), errors='coerce'
            )
            result['trade_date'] = pd.to_datetime(
                main_df['Trade_Date'].str.strip(), errors='coerce'
            )

            if is_cluster:
                result['insider_name'] = ''
                result['title'] = ''
                result['insider_count'] = pd.to_numeric(main_df.get('Ins', 0), errors='coerce').fillna(1).astype(int)
                result['industry'] = main_df.get('Industry', '')
            else:
                result['insider_name'] = main_df.get('Insider_Name', '')
                result['title'] = main_df.get('Title', '')
                result['insider_count'] = 1
                result['industry'] = ''

            result['trade_type'] = main_df['Trade_Type']
            
            # 解析价格和金额
            result['price'] = main_df['Price'].apply(self._parse_dollar)
            result['qty'] = main_df['Qty'].apply(self._parse_number)
            result['value'] = main_df['Value'].apply(self._parse_dollar)
            result['ownership_change'] = main_df.get('ΔOwn', '0%').apply(self._parse_pct)

            # 解析后续表现
            for col in ['1d', '1w', '1m', '6m']:
                if col in main_df.columns:
                    result[f'return_{col}'] = main_df[col].apply(self._parse_pct)
                else:
                    result[f'return_{col}'] = np.nan

            # 只保留买入 (P - Purchase)
            result = result[result['trade_type'].str.contains('P - Purchase', na=False)].copy()

            # 过滤有效 ticker
            result = result[result['ticker'].str.match(r'^[A-Z]{1,5}$', na=False)].copy()

            return result

        except Exception as e:
            logger.error(f"HTML 解析失败: {e}")
            return None

    @staticmethod
    def _parse_dollar(val) -> float:
        """解析美元金额 '$1,234.56' → 1234.56"""
        if pd.isna(val):
            return 0
        s = str(val).replace('$', '').replace(',', '').replace('+', '').replace('-', '').strip()
        try:
            return abs(float(s))
        except:
            return 0

    @staticmethod
    def _parse_number(val) -> int:
        """解析数量 '+4,835' → 4835"""
        if pd.isna(val):
            return 0
        s = str(val).replace(',', '').replace('+', '').replace('-', '').strip()
        try:
            return abs(int(float(s)))
        except:
            return 0

    @staticmethod
    def _parse_pct(val) -> float:
        """解析百分比 '+16%' → 16.0"""
        if pd.isna(val):
            return np.nan
        s = str(val).replace('%', '').replace('+', '').strip()
        try:
            return float(s)
        except:
            return np.nan


def fetch_insider_trades(use_cache: bool = True) -> pd.DataFrame:
    """一键获取内部人交易数据"""
    return InsiderFetcher().fetch_all(use_cache=use_cache)


if __name__ == "__main__":
    df = fetch_insider_trades(use_cache=False)
    print(f"\n总交易笔数: {len(df)}")
    if not df.empty:
        print(f"Ticker 数量: {df['ticker'].nunique()}")
        print(f"\n集群买入:")
        clusters = df[df['signal_type'] == 'cluster_buy']
        print(clusters[['ticker', 'insider_count', 'value', 'trade_date']].head(10).to_string())
