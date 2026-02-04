import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Load data
df = pd.read_csv('/home/mrakgr/Spiral-s-ML-Library/Spiral.Trading/data/new_generator_bars.csv')

# Use 1-second bars directly
bars = df.copy()

# Color mapping for trends
trend_colors = {
    'StrongUptrend': '#00AA00',
    'MidUptrend': '#66CC66',
    'WeakUptrend': '#99DD99',
    'Consolidation': '#888888',
    'WeakDowntrend': '#DD9999',
    'MidDowntrend': '#CC6666',
    'StrongDowntrend': '#AA0000'
}

fig, (ax, ax_vol) = plt.subplots(2, 1, figsize=(600, 30), height_ratios=[3, 1], sharex=True)

# Plot candlesticks
width = 0.6
for idx, row in bars.iterrows():
    x = row['Time']
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    color = trend_colors.get(row['Trend'], '#888888')
    
    # Body
    body_bottom = min(o, c)
    body_height = abs(c - o)
    rect = mpatches.Rectangle((x - width/2, body_bottom), width, body_height,
                               facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    
    # Wicks
    ax.plot([x, x], [l, body_bottom], color='black', linewidth=0.5)
    ax.plot([x, x], [body_bottom + body_height, h], color='black', linewidth=0.5)

ax.set_xlim(-5, len(bars) + 5)
ax.set_ylim(bars['Low'].min() * 0.995, bars['High'].max() * 1.005)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Price')
ax.set_title('New Order Flow Generator - 1-Second OHLC Bars (colored by trend)')

# Legend
legend_elements = [mpatches.Patch(facecolor=c, edgecolor='black', label=t) 
                   for t, c in trend_colors.items()]
ax.legend(handles=legend_elements, loc='upper left')

# Volume subplot
for idx, row in bars.iterrows():
    x = row['Time']
    vol = row['Volume']
    color = trend_colors.get(row['Trend'], '#888888')
    ax_vol.bar(x, vol, width=0.8, color=color, edgecolor='none')

ax_vol.set_ylabel('Volume')
ax_vol.set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig('/home/mrakgr/Spiral-s-ML-Library/Spiral.Trading/data/new_generator_ohlc.png', dpi=150)
print(f"Saved to /home/mrakgr/Spiral-s-ML-Library/Spiral.Trading/data/new_generator_ohlc.png")
plt.close()
