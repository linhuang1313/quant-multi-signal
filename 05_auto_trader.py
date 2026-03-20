"""
05 - 自动化量化交易机器人
===========================
结合 Alpaca API + 均线策略，实现自动化交易

工作流程：
1. 每次运行时获取最新行情
2. 计算技术指标（均线、RSI）
3. 根据策略信号自动下单
4. 记录交易日志

⚠️  这是一个教学 Demo，不构成投资建议
⚠️  使用前请先在 config.py 中配置 API Key

运行: python 05_auto_trader.py AAPL
"""

import sys
import json
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

from config import API_KEY, API_SECRET, SHORT_WINDOW, LONG_WINDOW


class QuantTrader:
    """量化交易机器人"""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.client = TradingClient(API_KEY, API_SECRET, paper=True)
        self.log_file = f"trade_log_{self.symbol}.json"
        self.trade_history = self._load_log()

    def _load_log(self) -> list:
        """加载交易日志"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_log(self, action: str, price: float, qty: float, reason: str):
        """保存交易日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "action": action,
            "price": price,
            "qty": qty,
            "reason": reason,
        }
        self.trade_history.append(entry)
        with open(self.log_file, 'w') as f:
            json.dump(self.trade_history, f, indent=2, ensure_ascii=False)
        print(f"📝 交易记录已保存到 {self.log_file}")

    def get_account(self):
        """获取账户信息"""
        return self.client.get_account()

    def get_position(self):
        """获取当前持仓"""
        try:
            pos = self.client.get_open_position(self.symbol)
            return {
                "qty": float(pos.qty),
                "avg_price": float(pos.avg_entry_price),
                "market_value": float(pos.market_value),
                "pnl": float(pos.unrealized_pl),
                "pnl_pct": float(pos.unrealized_plpc) * 100,
            }
        except APIError:
            return None

    def get_signals(self) -> dict:
        """
        获取当前技术指标和交易信号

        返回:
            dict: {
                "price": 当前价格,
                "sma_short": 短期均线,
                "sma_long": 长期均线,
                "rsi": RSI值,
                "signal": "BUY" / "SELL" / "HOLD",
                "reasons": [信号原因列表]
            }
        """
        # 获取历史数据（多取一些用于计算指标）
        df = yf.download(self.symbol, period="3mo", progress=False)
        if hasattr(df.columns, 'droplevel'):
            df.columns = df.columns.droplevel(1)

        if df.empty or len(df) < LONG_WINDOW:
            return {"signal": "HOLD", "reasons": ["数据不足"]}

        close = df['Close']
        current_price = float(close.iloc[-1])

        # 计算均线
        sma_short = float(close.rolling(SHORT_WINDOW).mean().iloc[-1])
        sma_long = float(close.rolling(LONG_WINDOW).mean().iloc[-1])
        prev_sma_short = float(close.rolling(SHORT_WINDOW).mean().iloc[-2])
        prev_sma_long = float(close.rolling(LONG_WINDOW).mean().iloc[-2])

        # 计算 RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])

        # 生成信号
        reasons = []
        buy_score = 0
        sell_score = 0

        # 信号1: 均线金叉/死叉
        if prev_sma_short <= prev_sma_long and sma_short > sma_long:
            reasons.append(f"🟢 金叉: SMA{SHORT_WINDOW}({sma_short:.2f}) 上穿 SMA{LONG_WINDOW}({sma_long:.2f})")
            buy_score += 2
        elif prev_sma_short >= prev_sma_long and sma_short < sma_long:
            reasons.append(f"🔴 死叉: SMA{SHORT_WINDOW}({sma_short:.2f}) 下穿 SMA{LONG_WINDOW}({sma_long:.2f})")
            sell_score += 2

        # 信号2: 均线方向
        if sma_short > sma_long:
            reasons.append(f"🟢 短均线在长均线上方（多头排列）")
            buy_score += 1
        else:
            reasons.append(f"🔴 短均线在长均线下方（空头排列）")
            sell_score += 1

        # 信号3: RSI
        if rsi < 30:
            reasons.append(f"🟢 RSI={rsi:.1f} 超卖区")
            buy_score += 1
        elif rsi > 70:
            reasons.append(f"🔴 RSI={rsi:.1f} 超买区")
            sell_score += 1
        else:
            reasons.append(f"⚪ RSI={rsi:.1f} 中性区")

        # 综合判断
        if buy_score >= 3:
            signal = "BUY"
        elif sell_score >= 3:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "price": current_price,
            "sma_short": sma_short,
            "sma_long": sma_long,
            "rsi": rsi,
            "signal": signal,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "reasons": reasons,
        }

    def execute_signal(self, signals: dict, max_position_pct: float = 0.2):
        """
        根据信号执行交易

        参数:
            signals: get_signals() 的返回值
            max_position_pct: 单只股票最大仓位占比（默认20%）
        """
        signal = signals["signal"]
        price = signals["price"]
        account = self.get_account()
        equity = float(account.equity)
        position = self.get_position()

        if signal == "BUY":
            if position and position["qty"] > 0:
                print(f"⏸️  已持有 {self.symbol} {position['qty']} 股，跳过买入")
                return

            # 计算买入数量（最多用 max_position_pct 的仓位）
            max_amount = equity * max_position_pct
            qty = int(max_amount / price)
            if qty <= 0:
                print(f"⚠️  资金不足，无法买入 {self.symbol}")
                return

            print(f"\n🟢 执行买入: {self.symbol} x{qty} @ ~${price:.2f}")
            order = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            result = self.client.submit_order(order_data=order)
            self._save_log("BUY", price, qty, "; ".join(signals["reasons"]))
            print(f"✅ 买入订单已提交 | 订单ID: {result.id}")

        elif signal == "SELL":
            if not position or position["qty"] <= 0:
                print(f"⏸️  未持有 {self.symbol}，跳过卖出")
                return

            qty = position["qty"]
            print(f"\n🔴 执行卖出: {self.symbol} x{int(qty)} @ ~${price:.2f}")
            order = MarketOrderRequest(
                symbol=self.symbol,
                qty=int(qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            result = self.client.submit_order(order_data=order)
            pnl = position["pnl"]
            self._save_log("SELL", price, int(qty),
                          f"盈亏: ${pnl:+,.2f} | " + "; ".join(signals["reasons"]))
            print(f"✅ 卖出订单已提交 | 盈亏: ${pnl:+,.2f} | 订单ID: {result.id}")

        else:
            print(f"⏸️  当前信号: HOLD，不操作")

    def run(self):
        """运行一次完整的分析和交易流程"""
        print(f"\n{'=' * 60}")
        print(f"🤖 量化交易机器人 - {self.symbol}")
        print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        # 1. 账户信息
        account = self.get_account()
        print(f"\n💰 账户: 权益 ${float(account.equity):,.2f} | "
              f"现金 ${float(account.cash):,.2f}")

        # 2. 当前持仓
        position = self.get_position()
        if position:
            emoji = "📈" if position["pnl"] >= 0 else "📉"
            print(f"{emoji} 持仓: {self.symbol} x{int(position['qty'])} | "
                  f"均价 ${position['avg_price']:.2f} | "
                  f"盈亏 ${position['pnl']:+,.2f} ({position['pnl_pct']:+.2f}%)")
        else:
            print(f"📦 当前未持有 {self.symbol}")

        # 3. 技术分析
        print(f"\n📊 技术分析:")
        signals = self.get_signals()
        print(f"   当前价格: ${signals['price']:.2f}")
        print(f"   SMA{SHORT_WINDOW}: ${signals['sma_short']:.2f}")
        print(f"   SMA{LONG_WINDOW}: ${signals['sma_long']:.2f}")
        print(f"   RSI(14): {signals['rsi']:.1f}")

        print(f"\n📡 信号分析:")
        for reason in signals["reasons"]:
            print(f"   {reason}")

        signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
        print(f"\n   ➡️  综合信号: {signal_emoji[signals['signal']]} {signals['signal']}")
        print(f"      (买入分 {signals['buy_score']} / 卖出分 {signals['sell_score']})")

        # 4. 执行交易
        self.execute_signal(signals)

        # 5. 交易历史
        if self.trade_history:
            print(f"\n📜 最近5笔交易:")
            for trade in self.trade_history[-5:]:
                print(f"   [{trade['timestamp'][:10]}] {trade['action']} "
                      f"{trade['symbol']} x{trade['qty']} @ ${trade['price']:.2f}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ 请先在 config.py 中配置你的 API Key!")
        print("   步骤:")
        print("   1. 登录 https://app.alpaca.markets")
        print("   2. 切换到 Paper Trading")
        print("   3. 点击 Generate New Key")
        print("   4. 将 Key 和 Secret 填入 config.py")
        sys.exit(1)

    trader = QuantTrader(symbol)
    trader.run()
