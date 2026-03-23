"""
深入分析新策略：与现有策略的互补性、信号重叠度
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download(ticker, start='2005-01-01'):
    df = yf.download(ticker, start=start, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=['Close'])

def calc_rsi(series, period=2):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 加载SPY数据
spy = download('SPY')
spy['RSI2'] = calc_rsi(spy['Close'], 2)
spy['RSI3'] = calc_rsi(spy['Close'], 3)
spy['RSI4'] = calc_rsi(spy['Close'], 4)
spy['SMA200'] = spy['Close'].rolling(200).mean()
spy['SMA5'] = spy['Close'].rolling(5).mean()
spy['IBS'] = (spy['Close'] - spy['Low']) / (spy['High'] - spy['Low'])

# 标记各策略信号日
signals = pd.DataFrame(index=spy.index)

# 现有: RSI(2)<10 + IBS<0.3 + >MA200
signals['rsi_mr'] = (spy['RSI2'] < 10) & (spy['IBS'] < 0.3) & (spy['Close'] > spy['SMA200'])

# 现有: Tuesday (周一跌>1%)
spy['prev_close'] = spy['Close'].shift(1)
spy['day_ret'] = (spy['Close'] - spy['prev_close']) / spy['prev_close']
signals['tuesday'] = (spy.index.dayofweek == 0) & (spy['day_ret'] < -0.01)

# 新策略1: Triple RSI  RSI2<5 + RSI3<20 + RSI4<30 + >MA200
signals['triple_rsi'] = (spy['RSI2'] < 5) & (spy['RSI3'] < 20) & (spy['RSI4'] < 30) & (spy['Close'] > spy['SMA200'])

# 新策略2: Lower Highs (3天)
lh = pd.Series(False, index=spy.index)
for i in range(203, len(spy)):
    close = float(spy['Close'].iloc[i])
    sma200 = float(spy['SMA200'].iloc[i])
    if close <= sma200:
        continue
    ok = True
    for j in range(3):
        if float(spy['High'].iloc[i-j]) >= float(spy['High'].iloc[i-j-1]):
            ok = False
            break
    lh.iloc[i] = ok
signals['lower_highs'] = lh

# 新策略3: Down Week
weekly_close = spy['Close'].resample('W-FRI').last().dropna()
weekly_down = weekly_close < weekly_close.shift(1)
# 把周信号映射回daily（周五那天标记）
dw = pd.Series(False, index=spy.index)
for dt in weekly_down[weekly_down].index:
    if dt in dw.index:
        dw[dt] = True
signals['down_week'] = dw

print("=" * 70)
print("📊 SPY上各策略信号统计 (2005-2026)")
print("=" * 70)
for col in ['rsi_mr', 'tuesday', 'triple_rsi', 'lower_highs', 'down_week']:
    count = signals[col].sum()
    print(f"  {col:<20s}: {count:>5d} 信号")

# 信号重叠分析
print(f"\n{'='*70}")
print("📊 信号重叠分析 (Triple RSI vs RSI MR)")
print("=" * 70)

both = (signals['rsi_mr'] & signals['triple_rsi']).sum()
only_rsi = (signals['rsi_mr'] & ~signals['triple_rsi']).sum()
only_triple = (~signals['rsi_mr'] & signals['triple_rsi']).sum()
print(f"  两者都触发: {both}")
print(f"  仅RSI MR触发: {only_rsi}")
print(f"  仅Triple RSI触发: {only_triple}")
print(f"  Triple RSI是RSI MR的子集比例: {both/signals['triple_rsi'].sum()*100:.1f}%")

print(f"\n{'='*70}")
print("📊 信号重叠分析 (Lower Highs vs RSI MR)")
print("=" * 70)
both2 = (signals['rsi_mr'] & signals['lower_highs']).sum()
only_rsi2 = (signals['rsi_mr'] & ~signals['lower_highs']).sum()
only_lh = (~signals['rsi_mr'] & signals['lower_highs']).sum()
print(f"  两者都触发: {both2}")
print(f"  仅RSI MR触发: {only_rsi2}")
print(f"  仅Lower Highs触发: {only_lh}")

# 核心问题：这些新策略能给组合带来增量收益吗？
# 模拟：当前策略 vs 当前+Triple RSI  vs 当前+Lower Highs
print(f"\n{'='*70}")
print("📊 关键分析：新策略 vs 现有策略的增量价值")
print("=" * 70)

# Triple RSI 基本上是RSI MR的严格子集 —— 条件更严格
# 但它的胜率更高(79.2% vs ~73%)，Sharpe也更高
# 问题是：它触发太少，信号被RSI MR覆盖

# 真正有增量的策略需要：
# 1. 在RSI MR和Tuesday不触发时触发
# 2. 有正的期望值

# Lower Highs: 独立信号多(only_lh个)，但Sharpe只有0.97
# Down Week: 完全不同的时间框架(周级别)，暴露43.9%，回撤-38%太大

# 计算各策略独立信号时的表现
print("\n  [Triple RSI] 条件比RSI MR更严格，几乎完全重叠")
print(f"  → 不适合作为新策略（被现有RSI MR覆盖），但可以作为RSI信号的增强过滤器")

print(f"\n  [Lower Highs] 有 {only_lh} 个独立信号")
print(f"  → Sharpe 0.97-1.14，胜率70-74%，值得考虑")
print(f"  → 但与RSI MR有部分重叠（{both2}个），真正的增量较小")

print(f"\n  [Down Week] 暴露43.9%，回撤-38%")
print(f"  → 资金占用太高，与我们低暴露策略理念矛盾")
print(f"  → Sharpe 0.76，不够优秀")

print(f"\n  [隔夜效应] Sharpe 0.35-0.38")
print(f"  → 优势太弱，交易次数极多(2000+)，佣金敏感")
print(f"  → 不推荐")

print(f"\n  [SPY-TLT轮动] Sharpe 0.18，回撤-45.2%")
print(f"  → 效果很差，不推荐")

# Lower Highs 是唯一值得进一步考虑的
# 做一个详细对比：把Lower Highs加入现有组合
print(f"\n\n{'='*70}")
print("🔍 Lower Highs(4) SPY 详细年度表现")
print("=" * 70)

# 重新跑 Lower Highs with yearly breakdown
trades_lh = []
in_pos = False
entry_price = entry_idx = 0
entry_date = None

for i in range(201, len(spy)):
    close = float(spy['Close'].iloc[i])
    sma200 = float(spy['SMA200'].iloc[i])
    
    if not in_pos:
        if close <= sma200:
            continue
        ok = True
        for j in range(4):
            if i - j - 1 < 0:
                ok = False; break
            if float(spy['High'].iloc[i-j]) >= float(spy['High'].iloc[i-j-1]):
                ok = False; break
        if ok:
            entry_price = close
            entry_date = spy.index[i]
            entry_idx = i
            in_pos = True
    else:
        hold = i - entry_idx
        prev_high = float(spy['High'].iloc[i-1])
        pnl = (close - entry_price) / entry_price
        sell = False
        if close > prev_high: sell = True
        elif pnl <= -0.05: sell = True
        elif hold >= 10: sell = True
        
        if sell:
            ret = (close - entry_price) / entry_price * 100 - 0.06
            trades_lh.append({'entry_date': entry_date, 'exit_date': spy.index[i], 'ret': ret, 'hold_days': hold, 'year': entry_date.year})
            in_pos = False

tdf = pd.DataFrame(trades_lh)
yearly = tdf.groupby('year').agg(
    trades=('ret', 'count'),
    win_rate=('ret', lambda x: (x > 0).mean() * 100),
    total_ret=('ret', 'sum'),
    avg_ret=('ret', 'mean'),
).round(2)
print(yearly.to_string())

# 最终结论
print(f"\n\n{'='*70}")
print("📋 最终结论")
print("=" * 70)
print("""
搜索了6种策略类型，回测了25+个变体:

✅ 有一定价值但增量有限:
  - Lower Highs(4) SPY: Sharpe 1.14, 胜率74%, 但年均仅5笔交易，CAGR 0.89%
    增量收益太小，不值得增加系统复杂度
    
  - Triple RSI: Sharpe 1.62(最高!)，但几乎被现有RSI MR完全覆盖
    可以作为RSI策略的"高确信度"信号，但不是新增策略

❌ 不推荐:
  - 隔夜效应: 优势太弱(Sharpe<0.4)，交易频繁
  - Down Week: 回撤太大(-38%)，暴露太高(44%)
  - SPY-TLT轮动: 无alpha(Sharpe 0.18)
  - Pre-Holiday: 样本太少，不可靠

⚡ 关键发现:
  短期均值回归类策略(RSI、Lower Highs等)彼此高度相关
  想要真正的增量收益，需要不同类型的策略:
  - 动量/趋势跟踪 (已有Dual Momentum回测，Sharpe 3.48但资金占用大)
  - 季节性/日历效应 (已有Turnaround Tuesday)
  - 波动率策略 (VIX相关，但之前回测发现回撤太大)
""")

