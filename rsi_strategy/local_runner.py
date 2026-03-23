"""
本地量化交易扫描器 (Windows)
============================
替代 Perplexity cron，在本地电脑上运行
- 美股开市期间每10分钟扫描一次持仓出场
- 收盘前30分钟扫描新信号+交易
- 自动判断交易日和交易时段
- 支持微信/Telegram/邮件通知（可选）

使用方法:
  1. pip install numpy pandas yfinance requests
  2. python local_runner.py

按 Ctrl+C 停止
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

# 确保能找到同目录的模块
sys.path.insert(0, str(Path(__file__).parent))

from rsi_trader import RSITrader
from stock_trader import StockTrader
from tuesday_trader import TuesdayTrader

# ============================================================
# 配置
# ============================================================
ET = ZoneInfo("America/New_York")
LOCAL_TZ = ZoneInfo("Asia/Singapore")  # 你的时区

# 扫描频率（分钟）
MONITOR_INTERVAL = 10       # 持仓监控：每10分钟
SIGNAL_BEFORE_CLOSE = 30    # 收盘前30分钟扫描信号

# 通知方式（可选，取消注释启用）
NOTIFY_METHOD = "console"   # "console" | "telegram" | "email"
# TELEGRAM_BOT_TOKEN = "你的bot token"
# TELEGRAM_CHAT_ID = "你的chat id"

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / 'data' / 'local_runner.log', encoding='utf-8'),
    ]
)
log = logging.getLogger(__name__)


def send_notification(title: str, body: str):
    """发送通知"""
    msg = f"\n{'='*50}\n📢 {title}\n{'='*50}\n{body}\n{'='*50}"
    log.info(msg)
    
    if NOTIFY_METHOD == "telegram":
        try:
            import requests
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": f"📢 {title}\n\n{body}"})
        except:
            pass
    elif NOTIFY_METHOD == "email":
        # 可以用 smtplib 发送邮件
        pass


def get_et_now():
    """获取美东时间"""
    return datetime.now(ET)


def is_market_hours():
    """判断是否在交易时段 (美东 9:30-16:00, 周一至周五)"""
    now = get_et_now()
    if now.weekday() >= 5:  # 周末
        return False, "weekend"
    
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    
    if now < market_open:
        return False, f"盘前 (开市 {market_open.strftime('%H:%M')} ET)"
    elif now > market_close:
        return False, f"收盘"
    else:
        return True, f"交易中 ({now.strftime('%H:%M')} ET)"


def is_near_close(minutes=30):
    """是否接近收盘"""
    now = get_et_now()
    close_time = now.replace(hour=16, minute=0, second=0)
    return 0 < (close_time - now).total_seconds() <= minutes * 60


def is_monday():
    return get_et_now().weekday() == 0


def is_tuesday():
    return get_et_now().weekday() == 1


# ============================================================
# 主循环
# ============================================================
def main():
    log.info("🚀 本地量化交易扫描器启动")
    log.info(f"   扫描频率: 每{MONITOR_INTERVAL}分钟")
    log.info(f"   信号扫描: 收盘前{SIGNAL_BEFORE_CLOSE}分钟")
    log.info(f"   通知方式: {NOTIFY_METHOD}")
    
    rsi_trader = RSITrader()
    stock_trader = StockTrader()
    tuesday_trader = TuesdayTrader()
    
    signal_scanned_today = False
    tuesday_buy_done = False
    tuesday_sell_done = False
    last_date = None
    
    while True:
        try:
            now_et = get_et_now()
            today = now_et.date()
            
            # 新的一天重置标志
            if today != last_date:
                signal_scanned_today = False
                tuesday_buy_done = False
                tuesday_sell_done = False
                last_date = today
                log.info(f"\n📅 新的一天: {today} ({now_et.strftime('%A')})")
            
            is_open, status = is_market_hours()
            
            if not is_open:
                # 非交易时段，等待较长时间
                local_now = datetime.now(LOCAL_TZ).strftime('%H:%M')
                log.info(f"💤 {status} | 本地 {local_now} | 等待5分钟...")
                time.sleep(300)  # 5分钟检查一次
                continue
            
            log.info(f"\n📊 {status}")
            
            # ── 1. 持仓监控 (每次循环都跑) ──
            log.info("── ETF 持仓监控 ──")
            try:
                exit_result = rsi_trader.check_exits()
                exits = exit_result.get('actions', [])
                if exits:
                    for e in exits:
                        send_notification(
                            f"⚡ RSI平仓: {e['symbol']}",
                            f"原因: {e['reason']}\n盈亏: {e.get('pnl_pct', 0):+.2f}% (${e.get('pnl_usd', 0):+.2f})\n持仓: {e.get('hold_days', 0)}天"
                        )
            except Exception as ex:
                log.error(f"ETF监控出错: {ex}")
            
            log.info("── 个股持仓监控 ──")
            try:
                stock_exit_result = stock_trader.check_exits_only()
                stock_exits = stock_exit_result.get('exits', [])
                if stock_exits:
                    for e in stock_exits:
                        send_notification(
                            f"📉 个股平仓: {e['symbol']}",
                            f"策略: {e.get('strategy', '')}\n原因: {e['reason']}\n盈亏: {e.get('pnl_pct', 0):+.2f}%\n持仓: {e.get('hold_days', 0)}天"
                        )
            except Exception as ex:
                log.error(f"个股监控出错: {ex}")
            
            # ── 2. 收盘前信号扫描 (每天只做一次) ──
            if is_near_close(SIGNAL_BEFORE_CLOSE) and not signal_scanned_today:
                log.info("\n🔔 收盘前信号扫描!")
                
                # ETF RSI 扫描
                try:
                    rsi_result = rsi_trader.scan_and_trade()
                    entries = rsi_result.get('entries', [])
                    for t in entries:
                        send_notification(
                            f"📈 RSI买入: {t['symbol']}",
                            f"{t['qty']}股 @ ${t['price']:.2f}\n原因: {t['reason']}"
                        )
                except Exception as ex:
                    log.error(f"ETF信号扫描出错: {ex}")
                
                # 个股扫描
                try:
                    stock_result = stock_trader.scan_and_trade()
                    entries = stock_result.get('entries', [])
                    for t in entries:
                        send_notification(
                            f"📈 个股买入: {t['symbol']}",
                            f"策略: {t.get('strategy', '')}\n{t['qty']}股 @ ${t['price']:.2f}\n原因: {t['reason']}"
                        )
                except Exception as ex:
                    log.error(f"个股信号扫描出错: {ex}")
                
                # Tuesday 策略
                if is_monday() and not tuesday_buy_done:
                    try:
                        tue_result = tuesday_trader.monday_scan_and_buy()
                        tuesday_buy_done = True
                        log.info(f"Tuesday周一扫描完成: {tue_result}")
                    except Exception as ex:
                        log.error(f"Tuesday买入出错: {ex}")
                
                if is_tuesday() and not tuesday_sell_done:
                    try:
                        tue_result = tuesday_trader.tuesday_sell()
                        tuesday_sell_done = True
                        log.info(f"Tuesday周二平仓完成: {tue_result}")
                    except Exception as ex:
                        log.error(f"Tuesday卖出出错: {ex}")
                
                signal_scanned_today = True
                log.info("✅ 今日信号扫描完成")
            
            # 等待下一次扫描
            log.info(f"⏳ 等待{MONITOR_INTERVAL}分钟...")
            time.sleep(MONITOR_INTERVAL * 60)
        
        except KeyboardInterrupt:
            log.info("\n⏹️ 用户中断，停止运行")
            break
        except Exception as ex:
            log.error(f"主循环异常: {ex}")
            time.sleep(60)  # 出错后等1分钟重试


if __name__ == '__main__':
    main()
