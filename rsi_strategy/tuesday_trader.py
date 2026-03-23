"""
Turnaround Tuesday 自动交易系统
================================
基于33年回测验证的周二反转效应

策略规则:
  SPY/QQQ: 周一收盘跌>1%(相比上一交易日) → 周一收盘买入 → 周二收盘卖出

回测表现:
  SPY: Sharpe 4.54 | 胜率 59.3% | 182笔/33年 | 暴露 2.2%
  QQQ: Sharpe 2.89 | 胜率 56.1% | 237笔/27年 | 暴露 3.4%

运行节奏:
  1. 周一收盘前(3:45pm ET): monday_scan_and_buy() — 检查周一是否跌1%, 是则买入
  2. 周二收盘前(3:45pm ET): tuesday_sell() — 卖出所有Tuesday持仓

Alpaca Paper Trading API
"""

import json
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

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

# ============================================================
# 策略参数
# ============================================================
TUESDAY_PARAMS = {
    'SPY': {
        'name': 'SPY Turnaround Tuesday',
        'drop_threshold': 0.01,    # 周一跌1%触发
        'position_pct': 0.40,      # 占账户40%仓位
        'sharpe_bt': 4.54,
        'win_rate_bt': 59.3,
    },
    'QQQ': {
        'name': 'QQQ Turnaround Tuesday',
        'drop_threshold': 0.01,    # 周一跌1%触发
        'position_pct': 0.40,      # 占账户40%仓位
        'sharpe_bt': 2.89,
        'win_rate_bt': 56.1,
    },
}


class TuesdayTrader:
    """Turnaround Tuesday 自动交易器"""
    
    def __init__(self):
        self.tracking_file = DATA_DIR / "tuesday_position_tracking.json"
        self.tracking = self._load_tracking()
        self.log_file = DATA_DIR / "tuesday_trade_log.json"
        self.trade_log = self._load_trade_log()
        self.et = ZoneInfo("America/New_York")
    
    def _load_tracking(self) -> Dict:
        if self.tracking_file.exists():
            with open(self.tracking_file) as f:
                return json.load(f)
        return {}
    
    def _save_tracking(self):
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking, f, indent=2, default=str)
    
    def _load_trade_log(self) -> List:
        if self.log_file.exists():
            with open(self.log_file) as f:
                return json.load(f)
        return []
    
    def _save_trade_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.trade_log, f, indent=2, default=str)
    
    def _get_et_now(self) -> datetime:
        """获取美东时间"""
        return datetime.now(self.et)
    
    # ── Alpaca API ──────────────────────────────
    
    def get_account(self) -> Optional[Dict]:
        resp = requests.get(f"{BASE_URL}/account", headers=HEADERS)
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"获取账户失败: {resp.status_code}")
        return None
    
    def get_positions(self) -> List[Dict]:
        resp = requests.get(f"{BASE_URL}/positions", headers=HEADERS)
        if resp.status_code == 200:
            return resp.json()
        return []
    
    def get_position(self, symbol) -> Optional[Dict]:
        resp = requests.get(f"{BASE_URL}/positions/{symbol}", headers=HEADERS)
        if resp.status_code == 200:
            return resp.json()
        return None
    
    def place_buy(self, symbol: str, qty: int, reason: str, entry_price: float = None) -> bool:
        """
        下单买入，同时挂 bracket order（止损+止盈）
        Tuesday策略: 止损-3%（持仓只有1天，止损更紧），止盈+2%
        """
        stop_loss_pct = 0.03    # Tuesday策略持仓短，止损更紧
        take_profit_pct = 0.02  # Tuesday目标收益适中
        
        if entry_price and entry_price > 0:
            stop_price = round(entry_price * (1 - stop_loss_pct), 2)
            take_profit_price = round(entry_price * (1 + take_profit_pct), 2)
            
            payload = {
                "symbol": symbol,
                "qty": str(qty),
                "side": "buy",
                "type": "market",
                "time_in_force": "gtc",
                "order_class": "bracket",
                "take_profit": {
                    "limit_price": str(take_profit_price)
                },
                "stop_loss": {
                    "stop_price": str(stop_price)
                }
            }
            bracket_info = f" [止损${stop_price} / 止盈${take_profit_price}]"
        else:
            payload = {
                "symbol": symbol,
                "qty": str(qty),
                "side": "buy",
                "type": "market",
                "time_in_force": "day"
            }
            bracket_info = " [无bracket]"
        
        resp = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload)
        if resp.status_code in (200, 201):
            print(f"  ✅ 买入 {symbol} {qty}股 — {reason}{bracket_info}")
            return True
        print(f"  ❌ 买入失败 {symbol}: {resp.text}")
        return False
    
    def cancel_open_orders(self, symbol: str):
        """取消某个标的的所有挂单"""
        resp = requests.get(f"{BASE_URL}/orders?status=open&symbols={symbol}", headers=HEADERS)
        if resp.status_code == 200:
            for order in resp.json():
                oid = order['id']
                requests.delete(f"{BASE_URL}/orders/{oid}", headers=HEADERS)
                print(f"    🗑️ 取消挂单 {symbol} (ID: {oid[:8]}...)")
    
    def place_sell(self, symbol: str, qty: str, reason: str) -> bool:
        self.cancel_open_orders(symbol)
        
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": "sell",
            "type": "market",
            "time_in_force": "day"
        }
        resp = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload)
        if resp.status_code in (200, 201):
            print(f"  ✅ 卖出 {symbol} {qty}股 — {reason}")
            return True
        print(f"  ❌ 卖出失败 {symbol}: {resp.text}")
        return False
    
    # ── 市场数据 ──────────────────────────────
    
    def _get_today_and_prev(self, ticker: str) -> Optional[Dict]:
        """
        获取今天和上一交易日的价格数据
        返回: {'today_close', 'today_open', 'prev_close', 'today_high', 'today_low'}
        """
        df = yf.download(ticker, period='5d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=['Close'])
        
        if len(df) < 2:
            return None
        
        today = df.iloc[-1]
        prev = df.iloc[-2]
        
        return {
            'today_close': float(today['Close']),
            'today_open': float(today['Open']),
            'today_high': float(today['High']),
            'today_low': float(today['Low']),
            'prev_close': float(prev['Close']),
            'today_date': str(df.index[-1].date()),
            'prev_date': str(df.index[-2].date()),
        }
    
    # ── 核心逻辑 ──────────────────────────────
    
    def monday_scan_and_buy(self) -> Dict:
        """
        周一收盘前运行:
        1. 确认今天是周一
        2. 检查SPY/QQQ是否跌>1%(相比上一交易日)
        3. 如果是 → 买入
        
        Returns:
            Dict: {'is_monday', 'signals', 'trades'}
        """
        et_now = self._get_et_now()
        weekday = et_now.weekday()  # 0=Monday
        
        print(f"\n{'='*65}")
        print(f"📅 Turnaround Tuesday — 周一信号检测 v1.0")
        print(f"   美东时间: {et_now.strftime('%Y-%m-%d %H:%M %A')}")
        print(f"{'='*65}")
        
        # 周一才执行买入逻辑
        if weekday != 0:
            print(f"\n  ⏭️ 今天不是周一 (weekday={weekday})，跳过买入扫描")
            return {'is_monday': False, 'signals': {}, 'trades': []}
        
        print(f"\n  ✅ 今天是周一，开始检测跌幅...")
        
        # 检查账户
        account = self.get_account()
        if not account:
            return {'is_monday': True, 'signals': {}, 'trades': [], 'error': '账户获取失败'}
        
        equity = float(account['equity'])
        cash = float(account['cash'])
        print(f"\n  💰 账户权益: ${equity:,.2f}  现金: ${cash:,.2f}")
        
        # 已有Tuesday持仓?
        current_positions = {p['symbol'] for p in self.get_positions()}
        
        signals = {}
        trades = []
        
        for ticker, params in TUESDAY_PARAMS.items():
            print(f"\n  📈 {ticker} ({params['name']}):")
            
            data = self._get_today_and_prev(ticker)
            if data is None:
                print(f"    ⚠️ 数据不足，跳过")
                continue
            
            today_close = data['today_close']
            prev_close = data['prev_close']
            drop_pct = (today_close - prev_close) / prev_close
            
            print(f"    上一交易日收盘: ${prev_close:.2f} ({data['prev_date']})")
            print(f"    今日(周一)收盘: ${today_close:.2f}")
            print(f"    跌幅: {drop_pct*100:+.2f}%  (阈值: -{params['drop_threshold']*100:.1f}%)")
            
            # 已有此ticker的Tuesday持仓?
            has_tuesday_pos = ticker in self.tracking
            already_in_position = ticker in current_positions
            
            # 信号判断: 跌幅 > 阈值
            dropped_enough = drop_pct <= -params['drop_threshold']
            
            if has_tuesday_pos:
                print(f"    ⏭️ 已有Tuesday持仓，跳过")
                signals[ticker] = {'signal': False, 'reason': '已有Tuesday持仓'}
                continue
            
            if dropped_enough:
                print(f"    🚀 触发买入信号! 周一跌{drop_pct*100:.2f}%")
                signals[ticker] = {
                    'signal': True,
                    'drop_pct': round(drop_pct * 100, 2),
                    'today_close': today_close,
                    'prev_close': prev_close,
                }
                
                # 执行买入
                # 注意: 如果RSI策略已占用仓位，需要考虑资金分配
                # Tuesday策略的仓位独立计算，但总资金有限
                target_value = equity * params['position_pct']
                
                if cash < target_value * 0.3:
                    print(f"    ⚠️ 现金不足 (${cash:,.2f} < ${target_value*0.3:,.2f})，跳过")
                    signals[ticker]['signal'] = False
                    signals[ticker]['reason'] = '现金不足'
                    continue
                
                # 不超过可用现金的90%
                buy_value = min(target_value, cash * 0.90)
                qty = max(1, int(buy_value / today_close))
                actual_value = qty * today_close
                
                if already_in_position:
                    # 如果RSI策略也持有此ticker，就不重复买入
                    print(f"    ⚠️ {ticker}已有RSI持仓，Tuesday跳过避免叠加")
                    signals[ticker]['signal'] = False
                    signals[ticker]['reason'] = 'RSI已持仓'
                    continue
                
                success = self.place_buy(ticker, qty, f"Tuesday买入: 周一跌{drop_pct*100:.2f}%", entry_price=today_close)
                
                if success:
                    self.tracking[ticker] = {
                        'entry_date': et_now.isoformat(),
                        'entry_price': today_close,
                        'qty': qty,
                        'drop_pct': round(drop_pct * 100, 2),
                        'strategy': 'turnaround_tuesday',
                    }
                    self._save_tracking()
                    
                    cash -= actual_value
                    trade = {
                        'action': 'BUY',
                        'symbol': ticker,
                        'qty': qty,
                        'price': today_close,
                        'value': actual_value,
                        'reason': f"Tuesday买入: 周一跌{drop_pct*100:.2f}%",
                        'time': et_now.isoformat(),
                        'strategy': 'turnaround_tuesday',
                    }
                    trades.append(trade)
                    self.trade_log.append(trade)
                    self._save_trade_log()
            else:
                print(f"    → 跌幅不足，无信号")
                signals[ticker] = {'signal': False, 'reason': f'跌幅{drop_pct*100:+.2f}%不足'}
        
        # 汇总
        print(f"\n{'='*65}")
        if trades:
            print(f"⚡ 执行了 {len(trades)} 笔Tuesday买入")
            for t in trades:
                print(f"   买入 {t['symbol']}: {t['qty']}股 ${t['price']:.2f}")
        else:
            print(f"✅ 周一无Tuesday买入信号")
        print(f"{'='*65}")
        
        return {
            'is_monday': True,
            'signals': signals,
            'trades': trades,
        }
    
    def tuesday_sell(self) -> Dict:
        """
        周二收盘前运行:
        卖出所有Tuesday策略持仓
        
        Returns:
            Dict: {'is_tuesday', 'actions'}
        """
        et_now = self._get_et_now()
        weekday = et_now.weekday()  # 1=Tuesday
        
        print(f"\n{'='*65}")
        print(f"📅 Turnaround Tuesday — 周二平仓 v1.0")
        print(f"   美东时间: {et_now.strftime('%Y-%m-%d %H:%M %A')}")
        print(f"{'='*65}")
        
        # 周二才执行卖出逻辑
        if weekday != 1:
            print(f"\n  ⏭️ 今天不是周二 (weekday={weekday})，跳过卖出")
            return {'is_tuesday': False, 'actions': []}
        
        # 检查是否有Tuesday持仓
        if not self.tracking:
            print(f"\n  ✅ 无Tuesday持仓，无需操作")
            return {'is_tuesday': True, 'actions': []}
        
        print(f"\n  📋 当前Tuesday持仓: {list(self.tracking.keys())}")
        
        actions = []
        positions = {p['symbol']: p for p in self.get_positions()}
        
        for ticker in list(self.tracking.keys()):
            track = self.tracking[ticker]
            entry_price = track['entry_price']
            qty = track['qty']
            
            pos = positions.get(ticker)
            if pos:
                current_price = float(pos['current_price'])
                unrealized_pl = float(pos['unrealized_pl'])
                unrealized_plpc = float(pos['unrealized_plpc'])
                actual_qty = str(int(float(pos['qty'])))
            else:
                # 可能已被RSI策略卖出或其他原因
                print(f"\n  ⚠️ {ticker} 无Alpaca持仓，清理tracking")
                del self.tracking[ticker]
                self._save_tracking()
                continue
            
            pnl_pct = unrealized_plpc * 100
            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            
            print(f"\n  {pnl_emoji} {ticker}  {actual_qty}股  "
                  f"买入价${entry_price:.2f}  现价${current_price:.2f}  "
                  f"盈亏{pnl_pct:+.2f}% (${unrealized_pl:+.2f})")
            
            reason = f"📅 Tuesday正常平仓: 盈亏{pnl_pct:+.2f}%"
            success = self.place_sell(ticker, actual_qty, reason)
            
            trade = {
                'action': 'SELL',
                'symbol': ticker,
                'qty': actual_qty,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_pct': round(pnl_pct, 2),
                'pnl_usd': round(unrealized_pl, 2),
                'reason': reason,
                'success': success,
                'time': et_now.isoformat(),
                'strategy': 'turnaround_tuesday',
            }
            actions.append(trade)
            self.trade_log.append(trade)
            self._save_trade_log()
            
            if success:
                del self.tracking[ticker]
                self._save_tracking()
        
        # 汇总
        print(f"\n{'='*65}")
        if actions:
            total_pnl = sum(a.get('pnl_usd', 0) for a in actions)
            print(f"⚡ Tuesday平仓 {len(actions)} 笔  总盈亏: ${total_pnl:+.2f}")
            for a in actions:
                print(f"   卖出 {a['symbol']}: {a['pnl_pct']:+.2f}% (${a['pnl_usd']:+.2f})")
        else:
            print(f"✅ 无Tuesday平仓操作")
        print(f"{'='*65}")
        
        return {'is_tuesday': True, 'actions': actions}
    
    def emergency_exit(self) -> Dict:
        """
        紧急平仓: 卖出所有Tuesday持仓（不限星期几）
        用于手动干预或异常情况
        """
        if not self.tracking:
            return {'status': 'no_positions'}
        
        positions = {p['symbol']: p for p in self.get_positions()}
        actions = []
        
        for ticker in list(self.tracking.keys()):
            pos = positions.get(ticker)
            if pos:
                actual_qty = str(int(float(pos['qty'])))
                success = self.place_sell(ticker, actual_qty, "🚨 Tuesday紧急平仓")
                if success and ticker in self.tracking:
                    del self.tracking[ticker]
            else:
                del self.tracking[ticker]
        
        self._save_tracking()
        return {'status': 'done', 'actions': actions}
    
    def get_status(self) -> Dict:
        """获取当前Tuesday策略状态"""
        et_now = self._get_et_now()
        return {
            'time_et': et_now.strftime('%Y-%m-%d %H:%M %A'),
            'weekday': et_now.weekday(),
            'active_positions': dict(self.tracking),
            'total_trades': len(self.trade_log),
        }


if __name__ == '__main__':
    trader = TuesdayTrader()
    
    et_now = datetime.now(ZoneInfo("America/New_York"))
    weekday = et_now.weekday()
    
    if weekday == 0:
        # 周一: 扫描买入
        result = trader.monday_scan_and_buy()
    elif weekday == 1:
        # 周二: 平仓
        result = trader.tuesday_sell()
    else:
        print(f"Turnaround Tuesday: 今天是 weekday={weekday}，非操作日")
        result = trader.get_status()
    
    print(f"\n结果: {json.dumps(result, indent=2, default=str)}")
