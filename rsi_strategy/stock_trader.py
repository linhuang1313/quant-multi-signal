"""
个股量化交易系统
========================
基于多策略×多指标评分系统，从498只美股中筛选的Top标的

策略:
  1. RSI(2) 均值回归: MA200上方 + RSI<阈值 → 买入, MA5上穿 → 卖出
  2. 布林带均值回归: MA200上方 + 跌破下轨 → 买入, 回到中轨 → 卖出

标的池 (综合评分Top，Sharpe>2, 股价<$600):
  RSI策略: CEG, IVZ, V, JCI, PKG, ROK, HIG, BSX, FIS, OTIS, DVN, POOL, MA, AAPL
  布林带策略: HWM, DD, NVDA, TXN, CSGP, POOL, DPZ

安全机制:
  - 财报前后3天不开新仓
  - OTO止损单 (券商实时监控)
  - 单只仓位上限30% ($600)
  - 同时最多持有2只个股
  - 与ETF策略共享账户但独立tracking

Alpaca Paper Trading API
"""

import json
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Alpaca API (与rsi_trader共享)
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
# 个股策略标的池 (来自多策略×多指标评分系统)
# ============================================================
STOCK_STRATEGIES = {
    # RSI均值回归策略标的 (综合评分Top, Sharpe>2)
    'CEG':  {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 89.7, 'sharpe': 4.10},
    'IVZ':  {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 89.4, 'sharpe': 3.80},
    'V':    {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 85.9, 'sharpe': 4.85},
    'JCI':  {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 85.6, 'sharpe': 4.13},
    'PKG':  {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 82.2, 'sharpe': 2.96},
    'ROK':  {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 81.2, 'sharpe': 3.24},
    'HIG':  {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 72.6, 'sharpe': 2.80},
    'BSX':  {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 78.0, 'sharpe': 2.77},
    'FIS':  {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 77.7, 'sharpe': 3.80},
    'DVN':  {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 74.0, 'sharpe': 2.48},
    'MA':   {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 73.7, 'sharpe': 2.69},
    'AAPL': {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 70.1, 'sharpe': 1.90},
    'OTIS': {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 75.1, 'sharpe': 2.48},
    'POOL': {'strategy': 'rsi', 'rsi_entry': 10, 'composite': 72.5, 'sharpe': 2.45},
    
    # 布林带均值回归策略标的
    'HWM':  {'strategy': 'bollinger', 'composite': 89.0, 'sharpe': 3.12},
    'DD':   {'strategy': 'bollinger', 'composite': 81.0, 'sharpe': 2.21},
    'NVDA': {'strategy': 'bollinger', 'composite': 80.6, 'sharpe': 4.44},
    'TXN':  {'strategy': 'bollinger', 'composite': 74.4, 'sharpe': 2.53},
    'CSGP': {'strategy': 'bollinger', 'composite': 73.1, 'sharpe': 2.01},  # MA_Pullback也好，这里用bollinger
    'DPZ':  {'strategy': 'bollinger', 'composite': 71.7, 'sharpe': 1.91},
}

# 资金管理参数
MAX_POSITION_PCT = 0.30      # 单只仓位上限30%
MAX_POSITIONS = 2            # 同时最多持有2只个股
STOCK_CAPITAL = 2000         # 分配给个股策略的资金
STOP_LOSS_PCT = 0.05         # 硬止损5%
MAX_HOLD_DAYS = 10           # 最大持仓天数
EARNINGS_BLACKOUT_DAYS = 3   # 财报前后N天不开仓


# ============================================================
# 技术指标
# ============================================================
def calc_rsi(series, period=2):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_market_data(ticker, days=260):
    """获取市场数据并计算全部指标"""
    df = yf.download(ticker, period=f'{days}d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None
    df = df.dropna(subset=['Close'])
    if len(df) < 201:
        return None
    
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA5'] = df['Close'].rolling(5).mean()
    
    # 布林带
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    
    return df


def check_earnings_blackout(ticker: str) -> Tuple[bool, Optional[str]]:
    """
    检查是否在财报黑名单期内
    Returns: (is_blocked, reason)
    """
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal and 'Earnings Date' in cal:
            earnings_dates = cal['Earnings Date']
            if not isinstance(earnings_dates, list):
                earnings_dates = [earnings_dates]
            
            today = date.today()
            for ed in earnings_dates:
                if isinstance(ed, datetime):
                    ed = ed.date()
                
                days_until = (ed - today).days
                
                # 财报前后N天内
                if -EARNINGS_BLACKOUT_DAYS <= days_until <= EARNINGS_BLACKOUT_DAYS:
                    return True, f"财报日{ed}，距今{days_until}天"
    except:
        pass
    
    return False, None


# ============================================================
# 信号检测
# ============================================================
def check_rsi_signal(df, ticker: str, params: dict) -> Optional[Dict]:
    """RSI均值回归信号检测"""
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi2 = float(latest['RSI2'])
    sma200 = float(latest['SMA200'])
    
    rsi_entry = params.get('rsi_entry', 10)
    
    if close > sma200 and rsi2 < rsi_entry:
        return {
            'signal': True,
            'strategy': 'RSI',
            'ticker': ticker,
            'close': close,
            'rsi2': rsi2,
            'reason': f"RSI买入: RSI(2)={rsi2:.1f} < {rsi_entry}",
        }
    return None


def check_bollinger_signal(df, ticker: str, params: dict) -> Optional[Dict]:
    """布林带均值回归信号检测"""
    latest = df.iloc[-1]
    close = float(latest['Close'])
    sma200 = float(latest['SMA200'])
    bb_lower = float(latest['BB_lower'])
    
    if close > sma200 and close < bb_lower:
        return {
            'signal': True,
            'strategy': 'Bollinger',
            'ticker': ticker,
            'close': close,
            'bb_lower': bb_lower,
            'reason': f"布林带买入: 价格${close:.2f} < 下轨${bb_lower:.2f}",
        }
    return None


def check_exit_signal(df, ticker: str, strategy: str) -> Optional[str]:
    """检查出场信号 (RSI和布林带共用部分 + 各自特有出场)"""
    latest = df.iloc[-1]
    close = float(latest['Close'])
    sma5 = float(latest['SMA5'])
    bb_mid = float(latest['BB_mid'])
    
    if strategy == 'RSI':
        if close > sma5:
            return f"RSI正常出场: 价格${close:.2f} > MA5 ${sma5:.2f}"
    elif strategy == 'Bollinger':
        if close > bb_mid:
            return f"布林带正常出场: 价格${close:.2f} > 中轨${bb_mid:.2f}"
    
    return None


# ============================================================
# 交易器
# ============================================================
class StockTrader:
    """个股量化交易器"""
    
    def __init__(self):
        self.tracking_file = DATA_DIR / "stock_position_tracking.json"
        self.tracking = self._load_json(self.tracking_file, {})
        self.log_file = DATA_DIR / "stock_trade_log.json"
        self.trade_log = self._load_json(self.log_file, [])
    
    def _load_json(self, path, default):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default
    
    def _save_tracking(self):
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking, f, indent=2, default=str)
    
    def _save_trade_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.trade_log, f, indent=2, default=str)
    
    # ── Alpaca API ──
    
    def get_account(self) -> Optional[Dict]:
        resp = requests.get(f"{BASE_URL}/account", headers=HEADERS)
        return resp.json() if resp.status_code == 200 else None
    
    def get_positions(self) -> List[Dict]:
        resp = requests.get(f"{BASE_URL}/positions", headers=HEADERS)
        return resp.json() if resp.status_code == 200 else []
    
    def cancel_open_orders(self, symbol: str):
        resp = requests.get(f"{BASE_URL}/orders?status=open&symbols={symbol}", headers=HEADERS)
        if resp.status_code == 200:
            for order in resp.json():
                oid = order['id']
                requests.delete(f"{BASE_URL}/orders/{oid}", headers=HEADERS)
                print(f"    🗑️ 取消挂单 {symbol} (ID: {oid[:8]}...)")
    
    def place_buy(self, symbol: str, qty: int, reason: str, entry_price: float) -> bool:
        """OTO单: 买入 + 自动挂止损"""
        stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
        
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "oto",
            "stop_loss": {
                "stop_price": str(stop_price)
            }
        }
        
        resp = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload)
        if resp.status_code in (200, 201):
            print(f"  ✅ 买入 {symbol} {qty}股 — {reason} [止损${stop_price}]")
            return True
        print(f"  ❌ 买入失败 {symbol}: {resp.text}")
        return False
    
    def place_sell(self, symbol: str, qty: str, reason: str) -> bool:
        self.cancel_open_orders(symbol)
        payload = {
            "symbol": symbol, "qty": qty, "side": "sell",
            "type": "market", "time_in_force": "day"
        }
        resp = requests.post(f"{BASE_URL}/orders", headers=HEADERS, json=payload)
        if resp.status_code in (200, 201):
            print(f"  ✅ 卖出 {symbol} {qty}股 — {reason}")
            return True
        print(f"  ❌ 卖出失败 {symbol}: {resp.text}")
        return False
    
    # ── 核心逻辑 ──
    
    def scan_and_trade(self) -> Dict:
        """完整扫描+交易流程"""
        now = datetime.now()
        print(f"\n{'='*65}")
        print(f"📊 个股量化交易系统 v1.0")
        print(f"   时间: {now.strftime('%Y-%m-%d %H:%M')}")
        print(f"   标的池: {len(STOCK_STRATEGIES)} 只 | 策略: RSI + 布林带")
        print(f"   资金: ${STOCK_CAPITAL} | 单只上限: {MAX_POSITION_PCT*100:.0f}%")
        print(f"{'='*65}")
        
        # 账户状态
        account = self.get_account()
        if not account:
            return {'error': '获取账户失败'}
        
        equity = float(account['equity'])
        cash = float(account['cash'])
        print(f"\n  💰 总权益: ${equity:,.2f}  现金: ${cash:,.2f}")
        
        # Step 1: 检查现有个股持仓出场
        exits = self._check_exits()
        
        # Step 2: 扫描新信号
        entries = self._scan_entries(cash)
        
        # 汇总
        total = len(exits) + len(entries)
        print(f"\n{'='*65}")
        if total > 0:
            print(f"⚡ 执行了 {total} 笔操作")
        else:
            print(f"✅ 无操作")
        print(f"{'='*65}")
        
        return {'exits': exits, 'entries': entries}
    
    def _check_exits(self) -> List[Dict]:
        """检查个股持仓出场"""
        positions = self.get_positions()
        stock_positions = [p for p in positions if p['symbol'] in STOCK_STRATEGIES]
        
        if not stock_positions:
            print(f"\n  📭 无个股持仓")
            return []
        
        now = datetime.now()
        actions = []
        
        print(f"\n  📋 个股持仓监控 ({len(stock_positions)} 只):")
        
        for pos in stock_positions:
            symbol = pos['symbol']
            qty = pos['qty']
            avg_price = float(pos['avg_entry_price'])
            current_price = float(pos['current_price'])
            unrealized_plpc = float(pos['unrealized_plpc'])
            unrealized_pl = float(pos['unrealized_pl'])
            
            track = self.tracking.get(symbol, {})
            strategy = track.get('strategy', 'RSI')
            entry_date = datetime.fromisoformat(track['entry_date']) if 'entry_date' in track else now
            hold_days = (now - entry_date).days
            
            emoji = "🟢" if unrealized_plpc >= 0 else "🔴"
            print(f"\n    {emoji} {symbol} ({strategy})  {qty}股  均价${avg_price:.2f}  "
                  f"现价${current_price:.2f}  {unrealized_plpc*100:+.2f}%  {hold_days}天")
            
            # 出场判断
            reason = None
            
            # 技术指标出场
            df = get_market_data(symbol, days=30)
            if df is not None:
                reason = check_exit_signal(df, symbol, strategy)
            
            # 硬止损 (OTO单已挂，这里做double check)
            if not reason and unrealized_plpc <= -STOP_LOSS_PCT:
                reason = f"🛑 硬止损: {unrealized_plpc*100:+.1f}%"
            
            # 时间止损
            if not reason and hold_days >= MAX_HOLD_DAYS:
                reason = f"⏰ 时间止损: {hold_days}天"
            
            if reason:
                print(f"      → {reason}")
                sell_qty = str(int(float(qty)))
                success = self.place_sell(symbol, sell_qty, reason)
                
                trade = {
                    'action': 'SELL', 'symbol': symbol, 'strategy': strategy,
                    'qty': sell_qty, 'price': current_price,
                    'pnl_pct': round(unrealized_plpc * 100, 2),
                    'pnl_usd': round(unrealized_pl, 2),
                    'reason': reason, 'hold_days': hold_days,
                    'time': now.isoformat(),
                }
                actions.append(trade)
                self.trade_log.append(trade)
                self._save_trade_log()
                
                if success and symbol in self.tracking:
                    del self.tracking[symbol]
                    self._save_tracking()
            else:
                print(f"      → 继续持有")
        
        return actions
    
    def _scan_entries(self, cash: float) -> List[Dict]:
        """扫描新入场信号"""
        # 当前个股持仓数
        positions = self.get_positions()
        stock_pos_symbols = {p['symbol'] for p in positions if p['symbol'] in STOCK_STRATEGIES}
        current_count = len(stock_pos_symbols)
        
        if current_count >= MAX_POSITIONS:
            print(f"\n  📊 已持有 {current_count}/{MAX_POSITIONS} 只个股，不再新开仓")
            return []
        
        slots = MAX_POSITIONS - current_count
        max_value = STOCK_CAPITAL * MAX_POSITION_PCT
        
        print(f"\n  🔍 扫描个股信号 (可开 {slots} 个新仓位, 单只上限${max_value:.0f}):")
        
        signals = []
        
        # 按综合评分排序扫描
        sorted_stocks = sorted(STOCK_STRATEGIES.items(), 
                               key=lambda x: x[1]['composite'], reverse=True)
        
        for ticker, params in sorted_stocks:
            if ticker in stock_pos_symbols:
                continue  # 已持仓
            
            # 财报黑名单检查
            blocked, block_reason = check_earnings_blackout(ticker)
            if blocked:
                print(f"    ⏭️ {ticker}: 财报期 ({block_reason})")
                continue
            
            # 获取数据
            df = get_market_data(ticker)
            if df is None:
                continue
            
            # 根据策略类型检测信号
            signal = None
            if params['strategy'] == 'rsi':
                signal = check_rsi_signal(df, ticker, params)
            elif params['strategy'] == 'bollinger':
                signal = check_bollinger_signal(df, ticker, params)
            
            if signal:
                signal['composite'] = params['composite']
                signals.append(signal)
                print(f"    🚀 {ticker}: {signal['reason']} (评分{params['composite']})")
            else:
                # 只打印RSI接近触发的
                if params['strategy'] == 'rsi':
                    latest = df.iloc[-1]
                    rsi2 = float(latest['RSI2'])
                    if rsi2 < 20:
                        print(f"    👀 {ticker}: RSI={rsi2:.1f} 接近触发")
        
        if not signals:
            print(f"    → 无信号")
            return []
        
        # 按综合评分排序，取top N
        signals.sort(key=lambda x: x['composite'], reverse=True)
        
        entries = []
        for sig in signals[:slots]:
            ticker = sig['ticker']
            price = sig['close']
            qty = max(1, int(max_value / price))
            actual_value = qty * price
            
            if actual_value > cash * 0.9:
                qty = max(1, int(cash * 0.9 / price))
                actual_value = qty * price
            
            if actual_value > cash:
                print(f"    ⚠️ 现金不足，跳过 {ticker}")
                continue
            
            success = self.place_buy(ticker, qty, sig['reason'], price)
            if success:
                self.tracking[ticker] = {
                    'entry_date': datetime.now().isoformat(),
                    'entry_price': price,
                    'qty': qty,
                    'strategy': sig['strategy'],
                    'reason': sig['reason'],
                    'composite_score': sig['composite'],
                }
                self._save_tracking()
                cash -= actual_value
                
                trade = {
                    'action': 'BUY', 'symbol': ticker, 'strategy': sig['strategy'],
                    'qty': qty, 'price': price, 'value': actual_value,
                    'reason': sig['reason'], 'time': datetime.now().isoformat(),
                }
                entries.append(trade)
                self.trade_log.append(trade)
                self._save_trade_log()
        
        return entries
    
    def check_exits_only(self) -> Dict:
        """仅检查出场 (盘中监控用)"""
        print(f"\n{'='*65}")
        print(f"📊 个股持仓监控 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*65}")
        
        exits = self._check_exits()
        return {'exits': exits}


if __name__ == '__main__':
    trader = StockTrader()
    result = trader.scan_and_trade()
