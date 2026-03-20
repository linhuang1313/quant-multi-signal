"""
自动交易模块 - Auto Trader
============================
将评分信号转化为实际 Alpaca 订单

功能:
1. 根据信号评分计算仓位大小
2. 提交限价单/市价单到 Alpaca
3. 风险检查 (仓位上限、止损、回撤暂停)
4. 交易日志和持仓管理
"""

import sys
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 添加父目录到路径以导入 config
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import API_KEY, API_SECRET, BASE_URL
except ImportError:
    API_KEY = os.getenv("ALPACA_API_KEY", "")
    API_SECRET = os.getenv("ALPACA_API_SECRET", "")
    BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

DATA_DIR = Path(__file__).parent / "data"
LOG_DIR = Path(__file__).parent / "logs"


class CongressTrader:
    """国会跟单自动交易器"""

    def __init__(self,
                 max_position_pct: float = 0.10,
                 max_sector_pct: float = 0.30,
                 max_drawdown_pct: float = 0.15,
                 stop_loss_pct: float = 0.08,
                 score_full_threshold: int = 80,
                 score_half_threshold: int = 60):
        """
        Args:
            max_position_pct: 单只股票最大仓位占比
            max_sector_pct: 单行业最大占比
            max_drawdown_pct: 最大回撤阈值 (超过则暂停交易)
            stop_loss_pct: 个股止损比例
            score_full_threshold: 满仓阈值分数
            score_half_threshold: 半仓阈值分数
        """
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.score_full = score_full_threshold
        self.score_half = score_half_threshold
        
        self._api = None
        self._trade_log: List[Dict] = []
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def api(self):
        """延迟初始化 Alpaca API"""
        if self._api is None:
            try:
                from alpaca.trading.client import TradingClient
                self._api = TradingClient(API_KEY, API_SECRET, paper=True)
                logger.info("Alpaca API 连接成功")
            except Exception as e:
                logger.error(f"Alpaca API 连接失败: {e}")
                raise
        return self._api

    def get_account_info(self) -> Dict:
        """获取账户信息"""
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'pnl_today': float(account.equity) - float(account.last_equity),
            'pnl_today_pct': (float(account.equity) - float(account.last_equity)) / float(account.last_equity) * 100 if float(account.last_equity) > 0 else 0,
        }

    def get_positions(self) -> pd.DataFrame:
        """获取当前持仓"""
        positions = self.api.get_all_positions()
        if not positions:
            return pd.DataFrame()
        
        records = []
        for pos in positions:
            records.append({
                'ticker': pos.symbol,
                'shares': int(pos.qty),
                'avg_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pnl': float(pos.unrealized_pl),
                'unrealized_pct': float(pos.unrealized_plpc) * 100,
                'side': pos.side,
            })
        
        return pd.DataFrame(records)

    def execute_signals(self, signals: pd.DataFrame, 
                        dry_run: bool = True) -> List[Dict]:
        """
        执行交易信号
        
        Args:
            signals: 评分后的信号 DataFrame
            dry_run: 模拟模式 (True=不实际下单，仅打印)
            
        Returns:
            list: 执行结果列表
        """
        if signals.empty:
            print("⚠️ 没有需要执行的信号")
            return []
        
        # 风险检查
        if not self._check_risk():
            return []
        
        account = self.get_account_info()
        positions = self.get_positions()
        current_tickers = set(positions['ticker'].tolist()) if not positions.empty else set()
        
        print(f"\n{'='*60}")
        print(f"🤖 国会跟单交易执行器")
        print(f"{'='*60}")
        print(f"  模式:      {'🔵 模拟 (DRY RUN)' if dry_run else '🔴 实盘交易'}")
        print(f"  账户权益:  ${account['equity']:,.2f}")
        print(f"  可用资金:  ${account['cash']:,.2f}")
        print(f"  当前持仓:  {len(current_tickers)} 只")
        print(f"  待执行信号:{len(signals)} 个")
        print()
        
        results = []
        
        for _, signal in signals.iterrows():
            ticker = signal['ticker']
            score = signal['total_score']
            
            # 跳过已持有的
            if ticker in current_tickers:
                print(f"  ⏭️ {ticker} - 已持有，跳过")
                continue
            
            # 计算仓位
            position_pct = self._calculate_position_size(score)
            position_value = account['equity'] * position_pct
            
            # 限制最大仓位
            max_value = account['equity'] * self.max_position_pct
            position_value = min(position_value, max_value)
            
            if position_value < 100:  # 最小交易金额
                print(f"  ⏭️ {ticker} - 仓位太小，跳过")
                continue
            
            result = {
                'ticker': ticker,
                'score': score,
                'position_pct': position_pct * 100,
                'position_value': position_value,
                'representative': signal['representative'],
                'amount_range': signal['amount'],
                'status': 'pending',
            }
            
            if dry_run:
                result['status'] = 'simulated'
                print(f"  🔵 {ticker:6s} | 评分 {score:.0f} | "
                      f"仓位 {position_pct*100:.1f}% (${position_value:,.0f}) | "
                      f"跟随 {signal['representative']}")
            else:
                try:
                    order_result = self._place_order(ticker, position_value)
                    result.update(order_result)
                    result['status'] = 'submitted'
                    print(f"  ✅ {ticker:6s} | 评分 {score:.0f} | "
                          f"{order_result.get('qty', '?')} 股 @ 市价 | "
                          f"订单ID: {order_result.get('order_id', 'N/A')}")
                except Exception as e:
                    result['status'] = 'failed'
                    result['error'] = str(e)
                    print(f"  ❌ {ticker:6s} | 下单失败: {e}")
            
            results.append(result)
            current_tickers.add(ticker)
        
        # 保存交易日志
        self._save_trade_log(results)
        
        print(f"\n📋 执行完毕: {sum(1 for r in results if r['status'] in ['submitted', 'simulated'])} 成功 / "
              f"{sum(1 for r in results if r['status'] == 'failed')} 失败")
        
        return results

    def check_stop_losses(self, dry_run: bool = True) -> List[Dict]:
        """
        检查并执行止损
        
        Args:
            dry_run: 模拟模式
            
        Returns:
            list: 止损执行结果
        """
        positions = self.get_positions()
        if positions.empty:
            print("✅ 无持仓需要检查")
            return []
        
        stop_results = []
        print(f"\n🛡️ 止损检查 (阈值: -{self.stop_loss_pct*100:.0f}%)")
        print("-" * 50)
        
        for _, pos in positions.iterrows():
            loss_pct = pos['unrealized_pct']
            
            if loss_pct <= -self.stop_loss_pct * 100:
                result = {
                    'ticker': pos['ticker'],
                    'loss_pct': loss_pct,
                    'shares': pos['shares'],
                    'unrealized_pnl': pos['unrealized_pnl'],
                }
                
                if dry_run:
                    result['status'] = 'would_sell'
                    print(f"  🔴 {pos['ticker']:6s} | 亏损 {loss_pct:.2f}% | "
                          f"${pos['unrealized_pnl']:,.2f} | [模拟] 将执行止损")
                else:
                    try:
                        self._close_position(pos['ticker'])
                        result['status'] = 'closed'
                        print(f"  🔴 {pos['ticker']:6s} | 亏损 {loss_pct:.2f}% | "
                              f"${pos['unrealized_pnl']:,.2f} | ✅ 已止损平仓")
                    except Exception as e:
                        result['status'] = 'failed'
                        result['error'] = str(e)
                        print(f"  ❌ {pos['ticker']:6s} | 止损失败: {e}")
                
                stop_results.append(result)
            else:
                status = "✅" if loss_pct >= 0 else "⚠️"
                print(f"  {status} {pos['ticker']:6s} | {loss_pct:+.2f}% | 安全")
        
        return stop_results

    def get_portfolio_summary(self) -> str:
        """生成投资组合摘要"""
        try:
            account = self.get_account_info()
            positions = self.get_positions()
        except Exception as e:
            return f"❌ 无法获取账户信息: {e}"
        
        lines = [
            "",
            "=" * 60,
            "💼 投资组合摘要",
            "=" * 60,
            "",
            "📊 账户概览:",
            f"  总权益:     ${account['equity']:>12,.2f}",
            f"  现金:       ${account['cash']:>12,.2f}",
            f"  购买力:     ${account['buying_power']:>12,.2f}",
            f"  今日盈亏:   ${account['pnl_today']:>12,.2f} ({account['pnl_today_pct']:+.2f}%)",
        ]
        
        if not positions.empty:
            total_value = positions['market_value'].sum()
            total_pnl = positions['unrealized_pnl'].sum()
            
            lines.extend([
                "",
                f"📈 持仓明细 ({len(positions)} 只):",
                f"  {'股票':6s} {'持股':>6s} {'均价':>10s} {'现价':>10s} {'市值':>12s} {'盈亏':>10s} {'盈亏%':>8s}",
                "  " + "-" * 68,
            ])
            
            for _, pos in positions.iterrows():
                pnl_symbol = "📈" if pos['unrealized_pct'] >= 0 else "📉"
                lines.append(
                    f"  {pos['ticker']:6s} {pos['shares']:>6d} "
                    f"${pos['avg_price']:>9.2f} ${pos['current_price']:>9.2f} "
                    f"${pos['market_value']:>11,.2f} "
                    f"${pos['unrealized_pnl']:>9,.2f} "
                    f"{pnl_symbol}{pos['unrealized_pct']:>+6.2f}%"
                )
            
            lines.extend([
                "  " + "-" * 68,
                f"  {'合计':6s} {'':>6s} {'':>10s} {'':>10s} "
                f"${total_value:>11,.2f} ${total_pnl:>9,.2f}",
            ])
        else:
            lines.append("\n  (无持仓)")
        
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _calculate_position_size(self, score: float) -> float:
        """根据信号评分计算仓位比例"""
        if score >= self.score_full:
            return self.max_position_pct  # 满仓 10%
        elif score >= self.score_half:
            return self.max_position_pct * 0.5  # 半仓 5%
        else:
            return 0

    def _check_risk(self) -> bool:
        """风险检查"""
        try:
            account = self.get_account_info()
            initial = 100_000  # Paper trading initial
            drawdown = (initial - account['equity']) / initial
            
            if drawdown >= self.max_drawdown_pct:
                print(f"🚫 风险警告: 回撤 {drawdown*100:.1f}% 超过阈值 {self.max_drawdown_pct*100:.0f}%")
                print("   交易已暂停，请手动检查投资组合")
                return False
            
            if account['cash'] < 1000:
                print(f"🚫 可用资金不足: ${account['cash']:,.2f}")
                return False
            
            return True
        except Exception as e:
            print(f"⚠️ 风险检查失败: {e}")
            return True  # 允许继续 (paper trading)

    def _place_order(self, ticker: str, position_value: float) -> Dict:
        """提交市价买入订单"""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        # 获取当前价格估算股数
        import yfinance as yf
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('regularMarketPrice') or stock.info.get('currentPrice', 0)
        
        if current_price <= 0:
            raise ValueError(f"无法获取 {ticker} 的当前价格")
        
        qty = int(position_value / current_price)
        if qty <= 0:
            raise ValueError(f"计算的股数为 0 (价格: ${current_price:.2f}, 仓位: ${position_value:.2f})")
        
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.api.submit_order(order_data)
        
        return {
            'order_id': str(order.id),
            'qty': qty,
            'estimated_price': current_price,
            'estimated_value': qty * current_price,
        }

    def _close_position(self, ticker: str) -> Dict:
        """平掉指定股票的全部仓位"""
        self.api.close_position(ticker)
        return {'status': 'closed', 'ticker': ticker}

    def _save_trade_log(self, results: List[Dict]):
        """保存交易日志"""
        log_file = LOG_DIR / f"trades_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(log_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'trades': results,
            }, f, indent=2, default=str)
        logger.info(f"交易日志已保存: {log_file}")


# ============================================================
# 便捷函数
# ============================================================

def execute_strategy(signals: pd.DataFrame, dry_run: bool = True) -> List[Dict]:
    """
    一键执行策略
    
    Args:
        signals: 评分后的信号
        dry_run: 模拟模式
        
    Returns:
        list: 执行结果
    """
    trader = CongressTrader()
    return trader.execute_signals(signals, dry_run=dry_run)


if __name__ == "__main__":
    trader = CongressTrader()
    print(trader.get_portfolio_summary())
