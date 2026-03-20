"""
数据获取模块 - Data Fetcher
============================
从多个免费数据源获取国会议员交易数据

数据源:
1. QuiverQuant 公开页面 (最近 300 笔交易，实时更新)
2. Senate Stock Watcher GitHub 存档 (历史数据 2015-2020)
"""

import re
import ast
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================
# 常量定义
# ============================================================

QUIVER_URL = "https://www.quiverquant.com/congresstrading/"
SENATE_GITHUB_URL = "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/aggregate/all_transactions.json"

# QuiverQuant 数据字段映射 (嵌入页面的列表索引)
QUIVER_FIELDS = [
    'ticker', 'asset_description', 'asset_type', 'trade_type', 'amount',
    'representative', 'chamber', 'party', 'report_date', 'transaction_date',
    'field11', 'trade_id', 'return_pct', 'full_name', 'photo_url', 'bioguide_id'
]

# 金额范围 → 中位数估值 (用于仓位评估)
AMOUNT_MIDPOINTS = {
    "$1,001 - $15,000": 8_000,
    "$15,001 - $50,000": 32_500,
    "$50,001 - $100,000": 75_000,
    "$100,001 - $250,000": 175_000,
    "$250,001 - $500,000": 375_000,
    "$500,001 - $1,000,000": 750_000,
    "$1,000,001 - $5,000,000": 3_000_000,
    "$5,000,001 - $25,000,000": 15_000_000,
    "$25,000,001 - $50,000,000": 37_500_000,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

DATA_DIR = Path(__file__).parent / "data"


class CongressDataFetcher:
    """国会交易数据获取器"""

    def __init__(self, cache_hours: int = 6):
        """
        Args:
            cache_hours: 数据缓存时间（小时），避免频繁请求
        """
        self.cache_hours = cache_hours
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_all(self, use_cache: bool = True) -> pd.DataFrame:
        """
        获取所有可用的国会交易数据
        
        Args:
            use_cache: 是否使用缓存数据
            
        Returns:
            pd.DataFrame: 标准化的交易数据
        """
        cache_file = DATA_DIR / "all_trades.csv"

        # 检查缓存
        if use_cache and cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(hours=self.cache_hours):
                logger.info("📦 使用缓存数据")
                df = pd.read_csv(cache_file, parse_dates=['transaction_date', 'report_date'])
                print(f"📦 从缓存加载 {len(df)} 笔交易记录")
                return df

        # 获取新数据
        print("🔄 正在从数据源获取最新数据...")
        dfs = []

        # 1. QuiverQuant 最近交易
        df_quiver = self._fetch_quiverquant()
        if df_quiver is not None and len(df_quiver) > 0:
            dfs.append(df_quiver)
            print(f"  ✅ QuiverQuant: {len(df_quiver)} 笔交易")

        # 2. Senate Stock Watcher 历史数据
        df_senate = self._fetch_senate_github()
        if df_senate is not None and len(df_senate) > 0:
            dfs.append(df_senate)
            print(f"  ✅ Senate历史: {len(df_senate)} 笔交易")

        if not dfs:
            print("❌ 无法获取任何数据源")
            return pd.DataFrame()

        # 合并、去重、排序
        df = pd.concat(dfs, ignore_index=True)
        df = self._clean_and_deduplicate(df)
        
        # 保存缓存
        df.to_csv(cache_file, index=False)
        print(f"\n📊 共获取 {len(df)} 笔有效交易记录")
        return df

    def fetch_recent(self, days: int = 30) -> pd.DataFrame:
        """
        仅获取最近 N 天的交易数据（用于实时信号）
        
        Args:
            days: 获取最近几天的数据
            
        Returns:
            pd.DataFrame: 最近的交易数据
        """
        df = self.fetch_all()
        if df.empty:
            return df
        
        cutoff = datetime.now() - timedelta(days=days)
        recent = df[df['transaction_date'] >= cutoff].copy()
        print(f"📅 最近 {days} 天交易: {len(recent)} 笔")
        return recent

    def _fetch_quiverquant(self) -> Optional[pd.DataFrame]:
        """从 QuiverQuant 公开页面提取嵌入的交易数据"""
        try:
            print("  🌐 正在获取 QuiverQuant 数据...")
            resp = requests.get(QUIVER_URL, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            # 提取嵌入的 JavaScript 数据
            match = re.search(r'let recentTradesData = (\[.*?\])\s*;', resp.text, re.DOTALL)
            if not match:
                logger.warning("QuiverQuant 页面结构已变更，无法提取数据")
                return None

            raw_data = ast.literal_eval(match.group(1))
            
            # 转换为 DataFrame
            records = []
            for row in raw_data:
                if len(row) < 16:
                    continue
                record = dict(zip(QUIVER_FIELDS, row))
                records.append(record)
            
            df = pd.DataFrame(records)
            
            # 标准化
            df = self._standardize_quiver(df)
            return df

        except Exception as e:
            logger.error(f"QuiverQuant 获取失败: {e}")
            print(f"  ⚠️ QuiverQuant 获取失败: {e}")
            return None

    def _fetch_senate_github(self) -> Optional[pd.DataFrame]:
        """从 GitHub 获取参议院历史交易数据"""
        try:
            cache_file = DATA_DIR / "senate_github_raw.json"
            
            # GitHub 数据不常更新，缓存 24 小时
            if cache_file.exists():
                mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - mod_time < timedelta(hours=24):
                    with open(cache_file) as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    return self._standardize_senate(df)
            
            print("  🌐 正在获取 Senate Stock Watcher 历史数据...")
            resp = requests.get(SENATE_GITHUB_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            # 缓存原始数据
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            df = pd.DataFrame(data)
            return self._standardize_senate(df)

        except Exception as e:
            logger.error(f"Senate GitHub 获取失败: {e}")
            print(f"  ⚠️ Senate GitHub 获取失败: {e}")
            return None

    def _standardize_quiver(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化 QuiverQuant 数据格式"""
        result = pd.DataFrame()
        result['ticker'] = df['ticker'].str.strip().str.upper()
        result['asset_description'] = df['asset_description']
        result['trade_type'] = df['trade_type'].map({
            'Purchase': 'Purchase',
            'Sale': 'Sale',
            'Sale (Full)': 'Sale',
            'Sale (Partial)': 'Sale',
        }).fillna(df['trade_type'])
        result['amount'] = df['amount']
        result['amount_est'] = df['amount'].map(AMOUNT_MIDPOINTS).fillna(8000)
        result['representative'] = df['representative']
        result['chamber'] = df['chamber']
        result['party'] = df['party']
        result['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        result['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
        result['bioguide_id'] = df['bioguide_id']
        result['source'] = 'quiverquant'
        result['return_since_trade'] = pd.to_numeric(df['return_pct'], errors='coerce')
        return result

    def _standardize_senate(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化 Senate Stock Watcher 数据格式"""
        # 过滤有效股票交易
        valid = df[
            (df['ticker'].notna()) & 
            (df['ticker'] != 'N/A') & 
            (df['ticker'] != '--') &
            (df['type'].isin(['Purchase', 'Sale (Full)', 'Sale (Partial)', 'Sale']))
        ].copy()

        result = pd.DataFrame()
        result['ticker'] = valid['ticker'].str.strip().str.upper()
        result['asset_description'] = valid['asset_description']
        result['trade_type'] = valid['type'].map({
            'Purchase': 'Purchase',
            'Sale': 'Sale',
            'Sale (Full)': 'Sale',
            'Sale (Partial)': 'Sale',
        }).fillna(valid['type'])
        result['amount'] = valid['amount']
        result['amount_est'] = valid['amount'].map(AMOUNT_MIDPOINTS).fillna(8000)
        result['representative'] = valid['senator']
        result['chamber'] = 'Senate'
        result['party'] = ''  # GitHub 数据没有 party 信息
        result['transaction_date'] = pd.to_datetime(valid['transaction_date'], format='%m/%d/%Y', errors='coerce')
        result['report_date'] = pd.NaT  # GitHub 数据没有 report_date
        result['bioguide_id'] = ''
        result['source'] = 'senate_github'
        result['return_since_trade'] = None
        return result

    def _clean_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗和去重数据"""
        # 移除无效 ticker
        df = df[df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != '--')].copy()
        
        # 移除非股票类型 (过滤掉基金份额等)
        # 保留常见股票 ticker 格式
        df = df[df['ticker'].str.match(r'^[A-Z]{1,5}$', na=False)].copy()
        
        # 按 ticker + representative + transaction_date 去重
        df = df.drop_duplicates(
            subset=['ticker', 'representative', 'transaction_date', 'trade_type'],
            keep='first'
        )
        
        # 按交易日期降序排列
        df = df.sort_values('transaction_date', ascending=False).reset_index(drop=True)
        
        # 计算申报延迟天数
        df['filing_delay_days'] = (df['report_date'] - df['transaction_date']).dt.days
        
        return df

    def get_summary(self, df: pd.DataFrame) -> str:
        """生成数据摘要"""
        if df.empty:
            return "无数据"
        
        lines = [
            "=" * 60,
            "📊 国会交易数据摘要",
            "=" * 60,
            f"总交易笔数:     {len(df):,}",
            f"时间范围:       {df['transaction_date'].min():%Y-%m-%d} 至 {df['transaction_date'].max():%Y-%m-%d}",
            f"涉及股票数:     {df['ticker'].nunique():,}",
            f"涉及议员数:     {df['representative'].nunique():,}",
            "",
            "📈 交易类型分布:",
        ]
        for ttype, count in df['trade_type'].value_counts().items():
            lines.append(f"  {ttype:15s}: {count:5d} ({count/len(df)*100:.1f}%)")
        
        lines.append("\n💰 金额分布:")
        for amt, count in df['amount'].value_counts().head(6).items():
            lines.append(f"  {amt:30s}: {count:5d}")
        
        lines.append("\n🏛️ 议院分布:")
        for chamber, count in df['chamber'].value_counts().items():
            lines.append(f"  {chamber:10s}: {count:5d}")
        
        lines.append("\n🔥 最活跃交易者 Top 10:")
        top_traders = df['representative'].value_counts().head(10)
        for name, count in top_traders.items():
            lines.append(f"  {name:30s}: {count:4d} 笔")
        
        lines.append("\n🎯 最受关注股票 Top 10:")
        top_stocks = df['ticker'].value_counts().head(10)
        for ticker, count in top_stocks.items():
            lines.append(f"  {ticker:6s}: {count:4d} 笔")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# 便捷函数
# ============================================================

def fetch_congress_trades(days: Optional[int] = None, use_cache: bool = True) -> pd.DataFrame:
    """
    一键获取国会交易数据
    
    Args:
        days: 仅获取最近 N 天的数据，None 表示获取全部
        use_cache: 是否使用缓存
        
    Returns:
        pd.DataFrame: 标准化的交易数据
    """
    fetcher = CongressDataFetcher()
    if days:
        return fetcher.fetch_recent(days)
    return fetcher.fetch_all(use_cache=use_cache)


if __name__ == "__main__":
    # 测试数据获取
    fetcher = CongressDataFetcher()
    df = fetcher.fetch_all(use_cache=False)
    print(fetcher.get_summary(df))
