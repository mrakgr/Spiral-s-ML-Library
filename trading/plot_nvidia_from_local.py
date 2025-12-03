import gzip
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# Directory with downloaded daily aggregates
data_dir = Path("data/daily_aggregates")

if not data_dir.exists():
    print(f"Error: Directory {data_dir} does not exist")
    print("Please run download_bulk_data.py first")
    exit(1)

print("="*60)
print("Extracting and plotting NVIDIA (NVDA) data from local files")
print("="*60)

# Get all .csv.gz files
files = sorted(data_dir.glob("*.csv.gz"))

if not files:
    print(f"No .csv.gz files found in {data_dir}")
    exit(1)

print(f"Found {len(files)} daily aggregate files")
print("Extracting NVDA data...")

# Extract NVDA data from all files
nvda_data = []

for file_path in files:
    try:
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f)
            
            # Filter for NVDA ticker
            if 'ticker' in df.columns:
                nvda_df = df[df['ticker'] == 'NVDA']
                if not nvda_df.empty:
                    nvda_data.append(nvda_df)
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")

if not nvda_data:
    print("No NVDA data found in downloaded files")
    exit(1)

# Combine all NVDA data
df = pd.concat(nvda_data, ignore_index=True)

# Convert timestamp to datetime
# The column is called 'window_start' in the flat files (nanoseconds)
if 'window_start' in df.columns:
    df['date'] = pd.to_datetime(df['window_start'], unit='ns')
elif 'timestamp' in df.columns:
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
else:
    print("Error: Neither 'window_start' nor 'timestamp' column found")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

print(f"\n✓ Extracted {len(df)} trading days for NVDA")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Apply split adjustments (flat files are NOT split-adjusted)
# NVDA splits: 2006-04-07 (2:1), 2007-09-11 (3:2), 2021-07-20 (4:1), 2024-06-10 (10:1)
print("\nApplying split adjustments...")

splits = [
    ('2006-04-07', 2.0),   # 1:2 split (2x)
    ('2007-09-11', 1.5),   # 2:3 split (1.5x)
    ('2021-07-20', 4.0),   # 1:4 split (4x)
    ('2024-06-10', 10.0),  # 1:10 split (10x)
]

# Calculate cumulative adjustment factor going forward from each split
df['adjustment_factor'] = 1.0

for split_date, ratio in splits:
    split_dt = pd.to_datetime(split_date)
    # For dates BEFORE the split, multiply by the ratio
    mask = df['date'] < split_dt
    df.loc[mask, 'adjustment_factor'] *= ratio

# Apply adjustment to price columns (divide) and volume (multiply by inverse)
price_cols = ['open', 'high', 'low', 'close']
for col in price_cols:
    df[col] = df[col] / df['adjustment_factor']

# Volume should be adjusted inversely (multiply by adjustment factor)
df['volume'] = df['volume'] * df['adjustment_factor']

print(f"✓ Applied {len(splits)} split adjustments")

# Filter out rows with missing data (handles weekends and holidays)
print("\nFiltering out missing data...")
initial_rows = len(df)
df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
# Also filter out rows where all OHLC values are zero
df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
filtered_rows = initial_rows - len(df)
if filtered_rows > 0:
    print(f"✓ Filtered out {filtered_rows} rows with missing/invalid data")

# Create interactive plot with plotly
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=('NVDA Price', 'Volume'),
    row_heights=[0.7, 0.3]
)

# Add candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='NVDA'
    ),
    row=1, col=1
)

# Add volume bar chart
fig.add_trace(
    go.Bar(
        x=df['date'],
        y=df['volume'],
        name='Volume',
        marker_color='rgba(0, 150, 255, 0.5)'
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    title='NVDA Stock Price and Volume (Split-Adjusted)',
    yaxis_title='Price (USD)',
    yaxis2_title='Volume',
    xaxis2_title='Date',
    height=800,
    xaxis_rangeslider_visible=False
)

# Export to HTML
output_html = Path("nvda_chart.html")
fig.write_html(str(output_html))
print(f"\n✓ Chart exported to: {output_html.absolute()}")

print("\n✓ Done!")
