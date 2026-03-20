"""
04 - Paper Trading 模拟盘下单
==============================
连接 Alpaca API 进行模拟盘交易
支持：市价单、限价单、止损单、查看持仓、取消订单

⚠️  使用前请先在 config.py 中配置你的 API Key

运行: python 04_paper_trading.py
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from config import API_KEY, API_SECRET
import sys


def create_client():
    """创建 Alpaca 交易客户端"""
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ 请先在 config.py 中配置你的 API Key!")
        print("   1. 登录 https://app.alpaca.markets")
        print("   2. 切换到 Paper Trading")
        print("   3. 点击 Generate New Key")
        print("   4. 将 Key 和 Secret 填入 config.py")
        sys.exit(1)
    return TradingClient(API_KEY, API_SECRET, paper=True)


def buy_market(client, symbol: str, qty: float = 1):
    """
    市价买入

    参数:
        symbol: 股票代码，如 AAPL
        qty: 数量（支持小数，即碎股）
    """
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
    order = client.submit_order(order_data=order_data)
    print(f"✅ 市价买入订单已提交!")
    print(f"   订单ID: {order.id}")
    print(f"   标的: {order.symbol}")
    print(f"   数量: {order.qty}")
    print(f"   状态: {order.status}")
    return order


def sell_market(client, symbol: str, qty: float = 1):
    """市价卖出"""
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    order = client.submit_order(order_data=order_data)
    print(f"✅ 市价卖出订单已提交!")
    print(f"   订单ID: {order.id}")
    print(f"   标的: {order.symbol}")
    print(f"   数量: {order.qty}")
    print(f"   状态: {order.status}")
    return order


def buy_limit(client, symbol: str, qty: float, limit_price: float):
    """
    限价买入
    只有当股价跌到 limit_price 或以下才会成交

    参数:
        symbol: 股票代码
        qty: 数量
        limit_price: 限价
    """
    order_data = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,  # Good Till Cancel
        limit_price=limit_price,
    )
    order = client.submit_order(order_data=order_data)
    print(f"✅ 限价买入订单已提交!")
    print(f"   标的: {order.symbol} | 限价: ${limit_price}")
    return order


def buy_stop_loss(client, symbol: str, qty: float, stop_price: float):
    """
    止损单
    当股价跌到 stop_price 时自动触发市价卖出

    参数:
        symbol: 股票代码
        qty: 数量
        stop_price: 止损价
    """
    order_data = StopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=stop_price,
    )
    order = client.submit_order(order_data=order_data)
    print(f"✅ 止损卖出订单已提交!")
    print(f"   标的: {order.symbol} | 止损价: ${stop_price}")
    return order


def view_orders(client):
    """查看所有挂单"""
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    orders = client.get_orders(filter=request)

    if not orders:
        print("📋 当前没有挂单")
        return

    print(f"\n📋 当前挂单 ({len(orders)} 个):")
    print("-" * 70)
    for o in orders:
        print(f"  [{o.order_type.upper()}] {o.side.upper()} {o.symbol} x{o.qty} | "
              f"状态: {o.status} | "
              f"提交时间: {o.submitted_at.strftime('%Y-%m-%d %H:%M')}")
    return orders


def view_positions(client):
    """查看所有持仓"""
    positions = client.get_all_positions()

    if not positions:
        print("📦 当前无持仓")
        return

    total_pnl = 0
    print(f"\n📦 当前持仓 ({len(positions)} 个):")
    print("-" * 70)
    for pos in positions:
        pnl = float(pos.unrealized_pl)
        total_pnl += pnl
        pnl_pct = float(pos.unrealized_plpc) * 100
        market_val = float(pos.market_value)
        emoji = "📈" if pnl >= 0 else "📉"
        print(f"  {emoji} {pos.symbol}: {pos.qty} 股 | "
              f"市值 ${market_val:,.2f} | "
              f"均价 ${float(pos.avg_entry_price):.2f} | "
              f"盈亏 ${pnl:+,.2f} ({pnl_pct:+.2f}%)")

    print("-" * 70)
    emoji = "🟢" if total_pnl >= 0 else "🔴"
    print(f"  {emoji} 总未实现盈亏: ${total_pnl:+,.2f}")


def cancel_all_orders(client):
    """取消所有挂单"""
    client.cancel_orders()
    print("✅ 已取消所有挂单")


def close_all_positions(client):
    """平掉所有持仓"""
    client.close_all_positions(cancel_orders=True)
    print("✅ 已提交平仓所有持仓的订单")


def interactive_menu():
    """交互式菜单"""
    client = create_client()

    while True:
        print(f"\n{'=' * 50}")
        print("🤖 Alpaca Paper Trading 操盘台")
        print("=" * 50)
        print("  1. 📊 查看账户信息")
        print("  2. 📦 查看持仓")
        print("  3. 📋 查看挂单")
        print("  4. 🟢 市价买入")
        print("  5. 🔴 市价卖出")
        print("  6. 💰 限价买入")
        print("  7. 🛡️  设置止损")
        print("  8. ❌ 取消所有挂单")
        print("  9. 🔄 平仓所有持仓")
        print("  0. 🚪 退出")
        print("-" * 50)

        choice = input("请选择操作 (0-9): ").strip()

        if choice == "0":
            print("👋 再见!")
            break

        elif choice == "1":
            account = client.get_account()
            print(f"\n  现金: ${float(account.cash):,.2f}")
            print(f"  总权益: ${float(account.equity):,.2f}")
            print(f"  购买力: ${float(account.buying_power):,.2f}")

        elif choice == "2":
            view_positions(client)

        elif choice == "3":
            view_orders(client)

        elif choice == "4":
            symbol = input("  股票代码 (如 AAPL): ").strip().upper()
            qty = float(input("  数量 (支持小数如 0.5): ").strip())
            buy_market(client, symbol, qty)

        elif choice == "5":
            symbol = input("  股票代码: ").strip().upper()
            qty = float(input("  数量: ").strip())
            sell_market(client, symbol, qty)

        elif choice == "6":
            symbol = input("  股票代码: ").strip().upper()
            qty = float(input("  数量: ").strip())
            price = float(input("  限价 ($): ").strip())
            buy_limit(client, symbol, qty, price)

        elif choice == "7":
            symbol = input("  股票代码: ").strip().upper()
            qty = float(input("  数量: ").strip())
            price = float(input("  止损价 ($): ").strip())
            buy_stop_loss(client, symbol, qty, price)

        elif choice == "8":
            cancel_all_orders(client)

        elif choice == "9":
            confirm = input("  确认平仓所有持仓? (y/n): ").strip().lower()
            if confirm == 'y':
                close_all_positions(client)

        else:
            print("❓ 无效选择，请重试")


if __name__ == "__main__":
    interactive_menu()
