"""
多信号共振量化策略 — 主控制台
================================
Multi-Signal Resonance Quantitative Strategy — Main Console

信号源:
1. 🏛️ 国会议员交易 (Congressional Trading)
2. 👔 内部人买入 (Insider Purchases via Form 4)
3. 📊 期权异动 (Unusual Options Activity)

用法:
    python main.py              # 交互式菜单
    python main.py --scan       # 快速扫描所有信号
    python main.py --backtest   # 运行回测
    python main.py --trade      # 模拟交易执行
    python main.py --portfolio  # 查看持仓
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import CongressDataFetcher, fetch_congress_trades
from insider_fetcher import InsiderFetcher, fetch_insider_trades
from options_fetcher import OptionsFetcher, scan_options_flow
from multi_signal import MultiSignalEngine
from signal_scorer import SignalScorer
from backtester import CongressBacktester
from trader import CongressTrader


def full_scan(lookback_days: int = 14, min_score: int = 30, top_n: int = 20):
    """全信号源扫描"""
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║" + "  🔮 多信号共振扫描".center(52) + "║")
    print("╚" + "═" * 60 + "╝")

    # 1. 获取三路数据
    congress_df = fetch_congress_trades(days=30)
    insider_df = fetch_insider_trades()
    options_df = scan_options_flow()

    # 2. 多信号共振分析
    engine = MultiSignalEngine(lookback_days=lookback_days, min_score=min_score)
    signals = engine.generate_signals(congress_df, insider_df, options_df, top_n=top_n)
    print(MultiSignalEngine.format_signals(signals))

    return signals


def congress_only_scan(days: int = 90, min_score: int = 45, top_n: int = 20):
    """仅国会信号扫描"""
    print("\n🏛️ 国会交易信号扫描")
    
    fetcher = CongressDataFetcher()
    df = fetcher.fetch_recent(days=days)
    
    if df.empty:
        print("❌ 无法获取数据")
        return None
    
    print(fetcher.get_summary(df))
    
    scorer = SignalScorer(min_score=min_score)
    signals = scorer.get_signals(df, top_n=top_n)
    print(scorer.format_signals(signals))
    
    return signals


def run_backtest(hold_days: int = 30, min_score: int = 35):
    """运行历史回测"""
    print("\n📊 运行回测")
    
    fetcher = CongressDataFetcher()
    df = fetcher.fetch_all()
    
    if df.empty:
        print("❌ 无法获取数据")
        return None
    
    scorer = SignalScorer(min_score=min_score)
    scored = scorer.score_trades(df)
    
    bt = CongressBacktester(hold_days=hold_days, min_signal_score=min_score)
    result = bt.run_backtest(scored)
    print(CongressBacktester.format_result(result))
    
    return result


def execute_trades(dry_run: bool = True, min_score: int = 30, top_n: int = 5):
    """基于多信号共振执行交易"""
    signals = full_scan(min_score=min_score, top_n=top_n)
    
    if signals is None or signals.empty:
        print("⚠️ 没有可执行的信号")
        return
    
    # 转换为 trader 兼容格式
    trade_signals = signals.copy()
    trade_signals['trade_type'] = 'Purchase'
    trade_signals['amount'] = ''
    trade_signals['representative'] = signals.get('sources', 'multi-signal')
    
    trader = CongressTrader()
    results = trader.execute_signals(trade_signals, dry_run=dry_run)
    trader.check_stop_losses(dry_run=dry_run)
    
    return results


def show_portfolio():
    """显示投资组合"""
    trader = CongressTrader()
    print(trader.get_portfolio_summary())


def interactive_menu():
    """交互式菜单"""
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + "  🔮 多信号共振量化交易系统 v2.0".center(50) + "║")
    print("║" + "  Multi-Signal Resonance Trading System".center(58) + "║")
    print("╚" + "═" * 60 + "╝")
    
    while True:
        print("\n📋 功能菜单:")
        print("  [1] 🔮 多信号共振扫描 (国会+内部人+期权)")
        print("  [2] 🏛️ 仅国会信号扫描")
        print("  [3] 📊 运行历史回测")
        print("  [4] 🤖 模拟交易执行 (DRY RUN)")
        print("  [5] 🔴 实盘交易执行 (真实下单)")
        print("  [6] 💼 查看投资组合")
        print("  [7] 🛡️ 检查止损")
        print("  [0] 退出")
        
        try:
            choice = input("\n请选择 [0-7]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if choice == '0':
            print("\n👋 再见!")
            break
        elif choice == '1':
            full_scan()
        elif choice == '2':
            congress_only_scan()
        elif choice == '3':
            try:
                days_input = input("持仓天数 [默认 30]: ").strip()
                hold_days = int(days_input) if days_input else 30
            except ValueError:
                hold_days = 30
            run_backtest(hold_days=hold_days)
        elif choice == '4':
            execute_trades(dry_run=True)
        elif choice == '5':
            confirm = input("⚠️ 确认执行实盘交易？(输入 YES 确认): ").strip()
            if confirm == 'YES':
                execute_trades(dry_run=False)
            else:
                print("已取消")
        elif choice == '6':
            show_portfolio()
        elif choice == '7':
            trader = CongressTrader()
            trader.check_stop_losses(dry_run=True)
        else:
            print("❌ 无效选择")


def main():
    parser = argparse.ArgumentParser(description="多信号共振量化交易系统")
    parser.add_argument('--scan', action='store_true', help='全信号源扫描')
    parser.add_argument('--congress', action='store_true', help='仅国会信号扫描')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--trade', action='store_true', help='模拟交易')
    parser.add_argument('--live', action='store_true', help='实盘交易')
    parser.add_argument('--portfolio', action='store_true', help='查看持仓')
    parser.add_argument('--days', type=int, default=14, help='信号回溯天数')
    parser.add_argument('--hold', type=int, default=30, help='回测持仓天数')
    parser.add_argument('--min-score', type=int, default=30, help='最低信号分数')
    
    args = parser.parse_args()
    
    if args.scan:
        full_scan(lookback_days=args.days, min_score=args.min_score)
    elif args.congress:
        congress_only_scan()
    elif args.backtest:
        run_backtest(hold_days=args.hold, min_score=args.min_score)
    elif args.trade:
        execute_trades(dry_run=True, min_score=args.min_score)
    elif args.live:
        execute_trades(dry_run=False, min_score=args.min_score)
    elif args.portfolio:
        show_portfolio()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
