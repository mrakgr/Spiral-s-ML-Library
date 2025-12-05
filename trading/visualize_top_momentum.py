"""
Visualize Top 20 Momentum Stocks for Latest Rebalancing Date

Shows the top performers by 6-month momentum along with their price charts.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

print("=" * 70)
print("TOP 20 MOMENTUM STOCKS - LATEST REBALANCING DATE")
print("=" * 70)

# Load the prepared momentum dataset
print("\nLoading momentum dataset...")
df = pd.read_parquet('data/momentum_prepared/momentum_data.parquet')
print(f"Loaded {len(df):,} rows")

# Load rebalancing dates
rebalance_dates_df = pd.read_csv('data/momentum_prepared/rebalance_dates.csv')
rebalance_dates_df['rebalance_date'] = pd.to_datetime(rebalance_dates_df['rebalance_date'])
rebalance_dates = rebalance_dates_df['rebalance_date'].tolist()

# Get latest rebalancing date
latest_rebalance = rebalance_dates[-1]
print(f"\nLatest rebalancing date: {latest_rebalance.date()}")

# Filter for latest rebalancing date and eligible stocks
latest_data = df[(df['date'] == latest_rebalance) & (df['is_eligible'])].copy()
print(f"Eligible stocks on this date: {len(latest_data):,}")

# Get top 20 by momentum rank
top_20 = latest_data.nsmallest(20, 'momentum_rank')[
    ['ticker', 'adj_close', 'momentum_6m', 'momentum_rank', 'avg_price_6m', 'avg_dollar_volume_1m']
].copy()

print(f"\n{'='*70}")
print(f"TOP 20 MOMENTUM STOCKS AS OF {latest_rebalance.date()}")
print(f"{'='*70}")
print(f"{'Rank':<6} {'Ticker':<8} {'Price':<10} {'6M Mom%':<12} {'Avg Price':<12} {'Avg $Vol':<15}")
print(f"{'-'*70}")

for idx, row in top_20.iterrows():
    rank = int(row['momentum_rank'])
    ticker = row['ticker']
    price = row['adj_close']
    momentum = row['momentum_6m'] * 100
    avg_price = row['avg_price_6m']
    avg_vol = row['avg_dollar_volume_1m']
    
    print(f"{rank:<6} {ticker:<8} ${price:<9.2f} {momentum:>10.2f}%  ${avg_price:<11.2f} ${avg_vol:>13,.0f}")

print(f"\n{'='*70}")

# Create charts for top 5
print("\nCreating charts for top 5 momentum stocks...")
top_5_tickers = top_20.head(5)['ticker'].tolist()

# Create subplot figure (5 rows, 1 column)
fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.05,
    subplot_titles=[f"{ticker} - 6M Momentum" for ticker in top_5_tickers],
    row_heights=[0.2, 0.2, 0.2, 0.2, 0.2]
)

for i, ticker in enumerate(top_5_tickers, 1):
    # Get data for this ticker (last 6 months for context)
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('date')
    
    # Get last 6 months
    last_6m = ticker_data[ticker_data['date'] >= (latest_rebalance - pd.Timedelta(days=180))]
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=last_6m['date'],
            open=last_6m['adj_open'],
            high=last_6m['adj_high'],
            low=last_6m['adj_low'],
            close=last_6m['adj_close'],
            name=ticker,
            showlegend=False
        ),
        row=i, col=1
    )
    
    # Get momentum value for annotation
    momentum_val = top_20[top_20['ticker'] == ticker]['momentum_6m'].values[0]
    
    # Update y-axis title for this subplot
    fig.update_yaxes(title_text=f"${ticker}", row=i, col=1)

# Update layout
fig.update_layout(
    title=f'Top 5 Momentum Stocks - {latest_rebalance.date()}',
    height=1400,
    showlegend=False,
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

# Remove rangesliders from all subplots
for i in range(1, 6):
    fig.update_xaxes(rangeslider_visible=False, row=i, col=1)

fig.show()

# Export to HTML
output_file = Path('top_momentum_charts.html')
fig.write_html(output_file)
print(f"\n✓ Charts exported to: {output_file}")

# Save top 20 list to CSV
output_csv = Path('data/momentum_prepared/top_20_latest.csv')
top_20.to_csv(output_csv, index=False)
print(f"✓ Top 20 list saved to: {output_csv}")

print("\n" + "=" * 70)
print("✓ Visualization complete!")
print("=" * 70)
