"""绘制组合回测权益曲线"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans']

# 加载数据
df = pd.read_csv('/home/user/workspace/quant-trading/rsi_strategy/data/combined_equity_curve.csv')
df['date'] = pd.to_datetime(df['date'])

fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1, 1]})
fig.suptitle('RSI Mean Reversion + Turnaround Tuesday | Combined Backtest (1999-2026)', 
             fontsize=16, fontweight='bold', y=0.98)

# ── 权益曲线 ──
ax1 = axes[0]
ax1.plot(df['date'], df['combined_equity'], color='#2196F3', linewidth=1.5, label='Combined Strategy')
ax1.plot(df['date'], df['spy_bh_equity'], color='#BDBDBD', linewidth=1, alpha=0.7, label='SPY Buy & Hold')
ax1.set_ylabel('Equity ($)', fontsize=12)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(df['date'].iloc[0], df['date'].iloc[-1])

# 指标文本框 - 放在右侧，避免和legend重叠
textstr = ('Combined: CAGR 5.61%  |  Sharpe 0.85  |  Max DD -12.1%\n'
           'SPY B&H: CAGR 7.61%  |  Max DD -55.2%\n'
           'Exposure: ~20%  |  752 trades over 27yr')
props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

# ── 回撤曲线 ──
ax2 = axes[1]
eq = df['combined_equity'].values
peak = np.maximum.accumulate(eq)
dd = (eq - peak) / peak * 100

spy_eq = df['spy_bh_equity'].values
spy_peak = np.maximum.accumulate(spy_eq)
spy_dd = (spy_eq - spy_peak) / spy_peak * 100

ax2.fill_between(df['date'], dd, 0, alpha=0.4, color='#2196F3', label='Combined DD')
ax2.fill_between(df['date'], spy_dd, 0, alpha=0.2, color='#9E9E9E', label='SPY DD')
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.legend(fontsize=10, loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(df['date'].iloc[0], df['date'].iloc[-1])
ax2.set_ylim(min(spy_dd.min(), dd.min()) * 1.1, 5)

# ── 年度收益对比 ──
ax3 = axes[2]
df['year'] = df['date'].dt.year
yearly = df.groupby('year').agg(
    combined_start=('combined_equity', 'first'),
    combined_end=('combined_equity', 'last'),
    spy_start=('spy_bh_equity', 'first'),
    spy_end=('spy_bh_equity', 'last'),
)
yearly['combined_ret'] = (yearly['combined_end'] / yearly['combined_start'] - 1) * 100
yearly['spy_ret'] = (yearly['spy_end'] / yearly['spy_start'] - 1) * 100

x = np.arange(len(yearly))
width = 0.35
ax3.bar(x - width/2, yearly['combined_ret'], width, label='Combined', color='#2196F3', alpha=0.8)
ax3.bar(x + width/2, yearly['spy_ret'], width, label='SPY B&H', color='#9E9E9E', alpha=0.6)
ax3.set_ylabel('Annual Return (%)', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(yearly.index, rotation=45, fontsize=8)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/user/workspace/quant-trading/rsi_strategy/data/combined_backtest_chart.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("Chart saved!")
