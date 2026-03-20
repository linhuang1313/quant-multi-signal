"""
持仓监控 + 自动止盈止损模块 - Portfolio Monitor
================================================
盘中每小时检查持仓，执行止盈止损策略

策略规则:
1. 固定止盈: +15% → 市价卖出
2. 固定止损: -8% → 市价卖出
3. 移动止损: 从持仓最高点回落 6% → 市价卖出
4. 最大持仓: 30天 → 到期平仓
5. 信号过期: 共振信号过期后降级关注

Alpaca Paper Trading API
"""

import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# 策略参数
# ============================================================
TAKE_PROFIT_PCT = 0.08       # 止盈 +8% (回测最优)
STOP_LOSS_PCT = -0.05        # 止损 -5% (回测最优)
TRAILING_STOP_PCT = 0.04     # 移动止损: 从高点回落 4% (回测最优)
MAX_HOLD_DAYS = 10           # 最大持仓 10 天 (回测最优)
POSITION_SIZE = 2000         # 每只股票默认仓位 $2,000

# ============================================================
# Alpaca API
# ============================================================
API_KEY = "PKCTMFDROQEWG5ESKZB75ZEPZA"
API_SECRET = "DynBRwaD34metqn1FsnTiPvCBxdBMKkRwoothbxwxWw8"
BASE_URL = "https://paper-api.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json"
}


class PortfolioMonitor:
    """持仓监控器"""

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.tracking_file = DATA_DIR / "position_tracking.json"
        self.tracking = self._load_tracking()

    def _load_tracking(self) -> Dict:
        """加载持仓追踪数据 (记录最高价、买入时间等)"""
        if self.tracking_file.exists():
            with open(self.tracking_file) as f:
                return json.load(f)
        return {}

    def _save_tracking(self):
        """保存追踪数据"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking, f, indent=2, default=str)

    def get_account(self) -> Optional[Dict]:
        """获取账户信息"""
        resp = requests.get(f"{BASE_URL}/account", headers=HEADERS)
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"获取账户失败: {resp.status_code}")
        return None

    def get_positions(self) -> List[Dict]:
        """获取当前持仓"""
        resp = requests.get(f"{BASE_URL}/positions", headers=HEADERS)
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"获取持仓失败: {resp.status_code}")
        return []

    def place_sell_order(self, symbol: str, qty: str, reason: str) -> bool:
        """提交卖出订单"""
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": "sell",
            "type": "market",
            "time_in_force": "day"
        }
        resp = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload)
        if resp.status_code in (200, 201):
            logger.info(f"✅ 卖出 {symbol} {qty}股 原因: {reason}")
            return True
        logger.error(f"❌ 卖出失败 {symbol}: {resp.text}")
        return False

    def place_buy_order(self, symbol: str, qty: int) -> bool:
        """提交买入订单"""
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        }
        resp = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload)
        if resp.status_code in (200, 201):
            logger.info(f"✅ 买入 {symbol} {qty}股")
            return True
        logger.error(f"❌ 买入失败 {symbol}: {resp.text}")
        return False

    def check_and_execute(self) -> Dict:
        """
        核心逻辑: 检查所有持仓，执行止盈止损
        
        Returns:
            Dict: 执行结果摘要
        """
        positions = self.get_positions()
        if not positions:
            return {"status": "no_positions", "actions": []}

        now = datetime.now()
        actions = []
        summary = {
            "check_time": now.isoformat(),
            "total_positions": len(positions),
            "total_market_value": 0,
            "total_pnl": 0,
            "actions": [],
        }

        print(f"\n{'='*65}")
        print(f"📊 持仓监控检查 - {now.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*65}")

        for pos in positions:
            symbol = pos['symbol']
            qty = pos['qty']
            avg_price = float(pos['avg_entry_price'])
            current_price = float(pos['current_price'])
            market_value = float(pos['market_value'])
            unrealized_pl = float(pos['unrealized_pl'])
            unrealized_plpc = float(pos['unrealized_plpc'])

            summary['total_market_value'] += market_value
            summary['total_pnl'] += unrealized_pl

            # 初始化追踪数据
            if symbol not in self.tracking:
                self.tracking[symbol] = {
                    'entry_date': now.isoformat(),
                    'entry_price': avg_price,
                    'high_price': current_price,
                    'high_date': now.isoformat(),
                }

            track = self.tracking[symbol]

            # 更新最高价
            if current_price > track.get('high_price', 0):
                track['high_price'] = current_price
                track['high_date'] = now.isoformat()

            high_price = track['high_price']
            drawdown_from_high = (current_price - high_price) / high_price if high_price > 0 else 0

            # 持仓天数
            entry_date = datetime.fromisoformat(track['entry_date'])
            hold_days = (now - entry_date).days

            # 状态显示
            pnl_emoji = "🟢" if unrealized_plpc >= 0 else "🔴"
            print(f"\n  {pnl_emoji} {symbol:6s}  {qty:>6s}股  均价${avg_price:.2f}  "
                  f"现价${current_price:.2f}  盈亏{unrealized_plpc*100:+.2f}%  "
                  f"高点回撤{drawdown_from_high*100:.1f}%  持仓{hold_days}天")

            # ============================================================
            # 止盈止损判断
            # ============================================================
            action = None
            reason = None

            # 1. 止盈检查
            if unrealized_plpc >= TAKE_PROFIT_PCT:
                action = "SELL"
                reason = f"🎯 止盈触发 ({unrealized_plpc*100:+.1f}% >= {TAKE_PROFIT_PCT*100}%)"

            # 2. 止损检查
            elif unrealized_plpc <= STOP_LOSS_PCT:
                action = "SELL"
                reason = f"🛑 止损触发 ({unrealized_plpc*100:+.1f}% <= {STOP_LOSS_PCT*100}%)"

            # 3. 移动止损检查 (需要先有盈利才启用)
            elif unrealized_plpc > 0.03 and drawdown_from_high <= -TRAILING_STOP_PCT:
                action = "SELL"
                reason = f"📉 移动止损 (从高点${high_price:.2f}回落{drawdown_from_high*100:.1f}%)"

            # 4. 最大持仓天数
            elif hold_days >= MAX_HOLD_DAYS:
                action = "SELL"
                reason = f"⏰ 持仓到期 ({hold_days}天 >= {MAX_HOLD_DAYS}天)"

            # 执行操作
            if action == "SELL":
                print(f"    → {reason}")
                # 取整数股数卖出
                sell_qty = str(int(float(qty)))
                success = self.place_sell_order(symbol, sell_qty, reason)
                actions.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "qty": sell_qty,
                    "reason": reason,
                    "pnl_pct": round(unrealized_plpc * 100, 2),
                    "pnl_usd": round(unrealized_pl, 2),
                    "success": success,
                })
                # 清除追踪
                if success and symbol in self.tracking:
                    del self.tracking[symbol]
            else:
                print(f"    → ✅ 继续持有")

        self._save_tracking()

        summary['actions'] = actions
        summary['total_pnl'] = round(summary['total_pnl'], 2)
        summary['total_market_value'] = round(summary['total_market_value'], 2)

        print(f"\n{'='*65}")
        print(f"💰 总市值: ${summary['total_market_value']:,.2f}  "
              f"总盈亏: ${summary['total_pnl']:+,.2f}")
        if actions:
            print(f"⚡ 执行了 {len(actions)} 笔交易")
        else:
            print(f"✅ 无触发条件，全部继续持有")
        print(f"{'='*65}")

        return summary

    def get_daily_report(self) -> str:
        """生成每日持仓报告"""
        account = self.get_account()
        positions = self.get_positions()
        
        if not account:
            return "无法获取账户信息"

        lines = [
            "",
            "=" * 60,
            f"📊 每日持仓报告 - {datetime.now().strftime('%Y-%m-%d')}",
            "=" * 60,
            f"💰 总权益: ${float(account['equity']):,.2f}",
            f"💵 现金:   ${float(account['cash']):,.2f}",
            f"📈 持仓数: {len(positions)} 只",
            "",
        ]

        if positions:
            total_pnl = 0
            total_mv = 0
            lines.append(f"{'股票':6s}  {'股数':>6s}  {'均价':>10s}  {'现价':>10s}  {'盈亏%':>8s}  {'盈亏$':>10s}")
            lines.append("-" * 60)
            
            for pos in sorted(positions, key=lambda x: float(x['unrealized_plpc']), reverse=True):
                sym = pos['symbol']
                qty = pos['qty']
                avg = float(pos['avg_entry_price'])
                cur = float(pos['current_price'])
                pnl = float(pos['unrealized_pl'])
                pnl_pct = float(pos['unrealized_plpc']) * 100
                mv = float(pos['market_value'])
                total_pnl += pnl
                total_mv += mv
                
                emoji = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"{emoji}{sym:5s}  {qty:>6s}  ${avg:>9.2f}  ${cur:>9.2f}  {pnl_pct:>+7.2f}%  ${pnl:>+9.2f}")
            
            lines.append("-" * 60)
            lines.append(f"{'合计':6s}  {'':>6s}  {'':>10s}  {'':>10s}  {'':>8s}  ${total_pnl:>+9.2f}")
            today_return = total_pnl / float(account['equity']) * 100 if float(account['equity']) > 0 else 0
            lines.append(f"\n📊 持仓总市值: ${total_mv:,.2f}")
            lines.append(f"📈 持仓收益率: {today_return:+.2f}%")
        else:
            lines.append("暂无持仓")

        lines.append("=" * 60)
        return "\n".join(lines)


def auto_trade_from_signals(signals_df, monitor: PortfolioMonitor) -> List[Dict]:
    """
    根据共振信号自动买入
    只买入双重/三重共振信号，且不重复建仓
    
    Args:
        signals_df: 共振引擎输出的信号 DataFrame
        monitor: PortfolioMonitor 实例
        
    Returns:
        List[Dict]: 买入操作记录
    """
    if signals_df is None or signals_df.empty:
        return []

    # 只自动买入双重/三重共振
    resonance = signals_df[signals_df['resonance_level'] >= 2].copy()
    if resonance.empty:
        return []

    # 获取当前持仓，避免重复
    current_positions = {p['symbol'] for p in monitor.get_positions()}
    
    # 获取账户现金
    account = monitor.get_account()
    if not account:
        return []
    available_cash = float(account['cash'])

    trades = []
    for _, row in resonance.iterrows():
        ticker = row['ticker']
        
        # 跳过已持仓
        if ticker in current_positions:
            print(f"  ⏭️ {ticker} 已持仓，跳过")
            continue

        # 检查资金是否足够
        if available_cash < POSITION_SIZE:
            print(f"  ⚠️ 现金不足 (${available_cash:,.0f} < ${POSITION_SIZE:,})")
            break

        # 计算股数
        import yfinance as yf
        try:
            stock = yf.Ticker(ticker)
            price = stock.fast_info.get('lastPrice', None)
            if not price or price <= 0:
                continue
            qty = max(1, int(POSITION_SIZE / price))
        except Exception:
            continue

        # 下单
        level = row['resonance_level']
        level_name = {2: "双重共振", 3: "三重共振"}.get(level, "共振")
        print(f"  🚀 自动买入 {ticker} {qty}股 (${price:.2f}) — {level_name} {row['total_score']}分")
        
        success = monitor.place_buy_order(ticker, qty)
        if success:
            available_cash -= qty * price
            current_positions.add(ticker)
            trades.append({
                "symbol": ticker,
                "qty": qty,
                "price": price,
                "reason": f"{level_name} {row['total_score']}分",
                "sources": row.get('sources', ''),
            })

    return trades


if __name__ == "__main__":
    monitor = PortfolioMonitor()
    
    # 检查持仓 + 止盈止损
    result = monitor.check_and_execute()
    
    # 打印日报
    print(monitor.get_daily_report())
