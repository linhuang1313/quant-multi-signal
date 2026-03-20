"""
01 - 账户连接测试
==================
验证 API Key 是否正确，查看账户信息和购买力

运行: python 01_account_check.py
"""

from alpaca.trading.client import TradingClient
from config import API_KEY, API_SECRET

def main():
    # 创建交易客户端（paper=True 表示模拟盘）
    client = TradingClient(API_KEY, API_SECRET, paper=True)

    # 获取账户信息
    account = client.get_account()

    print("=" * 50)
    print("📊 Alpaca 账户信息")
    print("=" * 50)
    print(f"  账户ID:     {account.id}")
    print(f"  账户状态:   {account.status}")
    print(f"  现金余额:   ${float(account.cash):,.2f}")
    print(f"  总权益:     ${float(account.equity):,.2f}")
    print(f"  购买力:     ${float(account.buying_power):,.2f}")
    print(f"  持仓市值:   ${float(account.long_market_value):,.2f}")
    print(f"  日内交易次数: {account.daytrade_count}")
    print("=" * 50)

    # 检查是否可以交易
    if account.trading_blocked:
        print("⚠️  账户交易被限制！")
    else:
        print("✅ 账户状态正常，可以交易")

    # 查看当前持仓
    positions = client.get_all_positions()
    if positions:
        print(f"\n📦 当前持仓 ({len(positions)} 个):")
        print("-" * 50)
        for pos in positions:
            pnl = float(pos.unrealized_pl)
            pnl_pct = float(pos.unrealized_plpc) * 100
            emoji = "📈" if pnl >= 0 else "📉"
            print(f"  {emoji} {pos.symbol}: {pos.qty} 股 | "
                  f"均价 ${float(pos.avg_entry_price):.2f} | "
                  f"盈亏 ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
    else:
        print("\n📦 当前无持仓")

if __name__ == "__main__":
    main()
