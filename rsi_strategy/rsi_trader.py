"""
RSI均值回归自动交易系统
========================
基于30+年回测验证的RSI(2)均值回归策略

策略规则:
  SPY: 价格>MA200 + RSI(2)<10 + IBS<0.3 → 买入, 价格>MA5 → 卖出
  QQQ: 价格>MA200 + RSI(2)<15 → 买入, 价格>MA5 → 卖出
  GLD: 价格>MA200 + RSI(2)<10 → 买入, 价格>MA5 → 卖出

回测表现:
  SPY: Sharpe 3.92 | 胜率 79.3% | 回撤 -14.0% | 33年217笔交易
  QQQ: Sharpe 2.21 | 胜率 69.3% | 回撤 -9.6% | 27年287笔交易
  GLD: Sharpe 1.08 | 胜率 73.4% | 回撤 -11.4% | 21年139笔交易

Alpaca Paper Trading API
"""

import json
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

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
STRATEGIES = {
    'SPY': {
        'name': 'SPY RSI均值回归',
        'rsi_period': 2,
        'rsi_entry': 10,       # RSI(2) < 10 买入
        'ma_period': 200,       # 价格须在MA200之上
        'ma_exit': 5,           # 价格 > MA5 卖出
        'use_ibs': True,        # 使用IBS过滤
        'ibs_threshold': 0.3,   # IBS < 0.3
        'position_pct': 0.40,   # 占账户40%仓位
        'max_hold_days': 10,    # 最大持仓10天（安全阀）
        'stop_loss_pct': 0.05,  # 硬止损5%
    },
    'QQQ': {
        'name': 'QQQ RSI均值回归',
        'rsi_period': 2,
        'rsi_entry': 15,       # RSI(2) < 15 买入（更宽松）
        'ma_period': 200,
        'ma_exit': 5,
        'use_ibs': False,       # QQQ不用IBS
        'ibs_threshold': 1.0,
        'position_pct': 0.40,   # 占账户40%仓位
        'max_hold_days': 10,
        'stop_loss_pct': 0.05,
    },
    'GLD': {
        'name': 'GLD 黄金RSI均值回归',
        'rsi_period': 2,
        'rsi_entry': 10,       # RSI(2) < 10 买入
        'ma_period': 200,       # 价格须在MA200之上
        'ma_exit': 5,           # 价格 > MA5 卖出
        'use_ibs': False,       # GLD不用IBS
        'ibs_threshold': 1.0,
        'position_pct': 0.20,   # 占账户20%仓位（黄金波动大，仓位小）
        'max_hold_days': 10,    # 最大持仓10天
        'stop_loss_pct': 0.05,  # 硬止损5%
    },
}

# ============================================================
# 技术指标计算
# ============================================================
def calc_rsi(series, period=2):
    """计算RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_ibs(high, low, close):
    """Internal Bar Strength"""
    rng = high - low
    if rng == 0:
        return 0.5
    return (close - low) / rng

def get_market_data(ticker, days=260):
    """获取市场数据并计算指标"""
    df = yf.download(ticker, period=f'{days}d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None
    
    # 去掉当天盘中NaN行
    df = df.dropna(subset=['Close'])
    if df.empty:
        return None
    
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    return df


class RSITrader:
    """RSI均值回归自动交易器"""
    
    def __init__(self):
        self.tracking_file = DATA_DIR / "rsi_position_tracking.json"
        self.tracking = self._load_tracking()
        self.log_file = DATA_DIR / "rsi_trade_log.json"
        self.trade_log = self._load_trade_log()
    
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
    
    def place_buy(self, symbol: str, qty: int, reason: str) -> bool:
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        }
        resp = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload)
        if resp.status_code in (200, 201):
            print(f"  ✅ 买入 {symbol} {qty}股 — {reason}")
            return True
        print(f"  ❌ 买入失败 {symbol}: {resp.text}")
        return False
    
    def place_sell(self, symbol: str, qty: str, reason: str) -> bool:
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
    
    # ── 核心逻辑 ──────────────────────────────
    
    def check_entry_signals(self) -> Dict:
        """
        检查RSI入场信号
        
        Returns:
            Dict: {ticker: {signal: bool, reason: str, data: {}}}
        """
        print(f"\n{'='*65}")
        print(f"📊 RSI均值回归信号检测 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*65}")
        
        signals = {}
        current_positions = {p['symbol'] for p in self.get_positions()}
        
        for ticker, params in STRATEGIES.items():
            print(f"\n  📈 {ticker} ({params['name']}):")
            
            df = get_market_data(ticker)
            if df is None or len(df) < 201:
                print(f"    ⚠️ 数据不足，跳过")
                continue
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            close = float(latest['Close'])
            rsi2 = float(latest['RSI2'])
            sma200 = float(latest['SMA200'])
            sma5 = float(latest['SMA5'])
            ibs = float(latest['IBS'])
            
            print(f"    价格: ${close:.2f}  RSI(2): {rsi2:.1f}  "
                  f"MA200: ${sma200:.2f}  MA5: ${sma5:.2f}  IBS: {ibs:.3f}")
            
            # 入场条件检查
            above_ma200 = close > sma200
            rsi_oversold = rsi2 < params['rsi_entry']
            ibs_ok = (not params['use_ibs']) or (ibs < params['ibs_threshold'])
            already_in = ticker in current_positions
            
            signal = above_ma200 and rsi_oversold and ibs_ok and not already_in
            
            status_parts = []
            status_parts.append(f"MA200 {'✅' if above_ma200 else '❌'}")
            status_parts.append(f"RSI2<{params['rsi_entry']} {'✅' if rsi_oversold else '❌'}({rsi2:.1f})")
            if params['use_ibs']:
                status_parts.append(f"IBS<{params['ibs_threshold']} {'✅' if ibs_ok else '❌'}({ibs:.3f})")
            if already_in:
                status_parts.append("已持仓 ⏭️")
            
            print(f"    条件: {' | '.join(status_parts)}")
            
            if signal:
                reason = f"RSI均值回归买入: RSI(2)={rsi2:.1f} (超卖)"
                print(f"    🚀 触发买入信号!")
                signals[ticker] = {
                    'signal': True,
                    'reason': reason,
                    'close': close,
                    'rsi2': rsi2,
                    'ibs': ibs,
                }
            else:
                print(f"    → 无信号")
                signals[ticker] = {'signal': False}
        
        return signals
    
    def execute_entries(self, signals: Dict) -> List[Dict]:
        """执行买入信号"""
        trades = []
        account = self.get_account()
        if not account:
            return trades
        
        equity = float(account['equity'])
        cash = float(account['cash'])
        
        for ticker, sig in signals.items():
            if not sig.get('signal'):
                continue
            
            params = STRATEGIES[ticker]
            target_value = equity * params['position_pct']
            
            if cash < target_value * 0.5:
                print(f"  ⚠️ 现金不足，跳过 {ticker}")
                continue
            
            price = sig['close']
            qty = max(1, int(target_value / price))
            actual_value = qty * price
            
            if actual_value > cash:
                qty = max(1, int(cash * 0.95 / price))
                actual_value = qty * price
            
            success = self.place_buy(ticker, qty, sig['reason'])
            if success:
                self.tracking[ticker] = {
                    'entry_date': datetime.now().isoformat(),
                    'entry_price': price,
                    'qty': qty,
                    'reason': sig['reason'],
                    'rsi_at_entry': sig['rsi2'],
                }
                self._save_tracking()
                
                cash -= actual_value
                trade = {
                    'action': 'BUY',
                    'symbol': ticker,
                    'qty': qty,
                    'price': price,
                    'value': actual_value,
                    'reason': sig['reason'],
                    'time': datetime.now().isoformat(),
                }
                trades.append(trade)
                self.trade_log.append(trade)
                self._save_trade_log()
        
        return trades
    
    def check_exits(self) -> Dict:
        """
        检查出场条件:
        1. 价格 > MA5 (正常出场)
        2. 亏损 > stop_loss (硬止损)
        3. 持仓 > max_hold_days (时间止损)
        
        Returns:
            Dict: 执行结果
        """
        positions = self.get_positions()
        rsi_positions = [p for p in positions if p['symbol'] in STRATEGIES]
        
        if not rsi_positions:
            return {"status": "no_rsi_positions", "actions": []}
        
        now = datetime.now()
        actions = []
        
        print(f"\n{'='*65}")
        print(f"📊 RSI持仓监控 — {now.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*65}")
        
        for pos in rsi_positions:
            symbol = pos['symbol']
            qty = pos['qty']
            avg_price = float(pos['avg_entry_price'])
            current_price = float(pos['current_price'])
            unrealized_pl = float(pos['unrealized_pl'])
            unrealized_plpc = float(pos['unrealized_plpc'])
            params = STRATEGIES[symbol]
            
            # 获取技术指标
            df = get_market_data(symbol, days=20)
            sma5 = None
            if df is not None and len(df) >= 5:
                sma5 = float(df['SMA5'].iloc[-1])
            
            # 持仓天数
            track = self.tracking.get(symbol, {})
            entry_date_str = track.get('entry_date', now.isoformat())
            entry_date = datetime.fromisoformat(entry_date_str)
            hold_days = (now - entry_date).days
            
            pnl_emoji = "🟢" if unrealized_plpc >= 0 else "🔴"
            print(f"\n  {pnl_emoji} {symbol}  {qty}股  均价${avg_price:.2f}  "
                  f"现价${current_price:.2f}  盈亏{unrealized_plpc*100:+.2f}%  "
                  f"持仓{hold_days}天")
            if sma5:
                print(f"    MA5=${sma5:.2f}  {'价格>MA5 ✅' if current_price > sma5 else '价格<MA5'}")
            
            # 出场判断
            action = None
            reason = None
            
            # 1. 正常出场: 价格 > MA5
            if sma5 and current_price > sma5:
                action = "SELL"
                reason = f"📈 RSI正常出场: 价格${current_price:.2f} > MA5 ${sma5:.2f}"
            
            # 2. 硬止损
            elif unrealized_plpc <= -params['stop_loss_pct']:
                action = "SELL"
                reason = f"🛑 硬止损: {unrealized_plpc*100:+.1f}% <= -{params['stop_loss_pct']*100}%"
            
            # 3. 时间止损
            elif hold_days >= params['max_hold_days']:
                action = "SELL"
                reason = f"⏰ 时间止损: {hold_days}天 >= {params['max_hold_days']}天"
            
            if action == "SELL":
                print(f"    → {reason}")
                sell_qty = str(int(float(qty)))
                success = self.place_sell(symbol, sell_qty, reason)
                
                trade = {
                    'action': 'SELL',
                    'symbol': symbol,
                    'qty': sell_qty,
                    'price': current_price,
                    'pnl_pct': round(unrealized_plpc * 100, 2),
                    'pnl_usd': round(unrealized_pl, 2),
                    'reason': reason,
                    'hold_days': hold_days,
                    'success': success,
                    'time': now.isoformat(),
                }
                actions.append(trade)
                self.trade_log.append(trade)
                self._save_trade_log()
                
                if success and symbol in self.tracking:
                    del self.tracking[symbol]
                    self._save_tracking()
            else:
                print(f"    → ✅ 继续持有 (等待价格>MA5)")
        
        return {"status": "checked", "actions": actions}
    
    def scan_and_trade(self) -> Dict:
        """
        完整扫描+交易流程:
        1. 检查出场条件
        2. 检查入场信号
        3. 执行交易
        """
        print(f"\n{'='*65}")
        print(f"🔄 RSI均值回归自动交易系统 v1.0")
        print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*65}")
        
        # 账户状态
        account = self.get_account()
        if account:
            print(f"\n  💰 账户权益: ${float(account['equity']):,.2f}")
            print(f"  💵 可用现金: ${float(account['cash']):,.2f}")
        
        # Step 1: 检查现有持仓出场
        exit_result = self.check_exits()
        
        # Step 2: 检查新入场信号
        signals = self.check_entry_signals()
        
        # Step 3: 执行买入
        entry_trades = self.execute_entries(signals)
        
        # 汇总
        exit_actions = exit_result.get('actions', [])
        total_actions = len(exit_actions) + len(entry_trades)
        
        print(f"\n{'='*65}")
        if total_actions > 0:
            print(f"⚡ 执行了 {total_actions} 笔操作")
            for a in exit_actions:
                print(f"   卖出 {a['symbol']}: {a['reason']} ({a.get('pnl_pct', 0):+.2f}%)")
            for t in entry_trades:
                print(f"   买入 {t['symbol']}: {t['qty']}股 ${t['price']:.2f}")
        else:
            print(f"✅ 无操作")
        print(f"{'='*65}")
        
        return {
            'exits': exit_actions,
            'entries': entry_trades,
            'signals': {k: v.get('signal', False) for k, v in signals.items()},
        }


def close_all_old_positions():
    """清仓旧策略的所有非RSI持仓"""
    resp = requests.get(f"{BASE_URL}/positions", headers=HEADERS)
    if resp.status_code != 200:
        print("获取持仓失败")
        return []
    
    positions = resp.json()
    rsi_tickers = set(STRATEGIES.keys())
    old_positions = [p for p in positions if p['symbol'] not in rsi_tickers]
    
    if not old_positions:
        print("无旧策略持仓需要清理")
        return []
    
    results = []
    print(f"\n🧹 清仓旧策略持仓 ({len(old_positions)} 只):")
    for pos in old_positions:
        symbol = pos['symbol']
        qty = str(int(float(pos['qty'])))
        pnl = float(pos['unrealized_pl'])
        pnl_pct = float(pos['unrealized_plpc']) * 100
        
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": "sell",
            "type": "market",
            "time_in_force": "day"
        }
        resp = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload)
        success = resp.status_code in (200, 201)
        emoji = "✅" if success else "❌"
        print(f"  {emoji} 卖出 {symbol} {qty}股  盈亏: {pnl_pct:+.2f}% (${pnl:+.2f})")
        results.append({
            'symbol': symbol,
            'qty': qty,
            'pnl_pct': round(pnl_pct, 2),
            'pnl_usd': round(pnl, 2),
            'success': success,
        })
    
    return results


if __name__ == '__main__':
    trader = RSITrader()
    result = trader.scan_and_trade()
