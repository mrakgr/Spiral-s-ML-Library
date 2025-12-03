import json
import pandas as pd
import mplfinance as mpf
from massive import RESTClient

with open("api_key.json") as f:
    api_keys = json.load(f)

client = RESTClient(api_keys["massive_api_key"])

aggs = []
for a in client.list_aggs(
    "NVDA",
    1,
    "minute",
    "2025-12-01",
    "2025-12-03",
    limit=50000,
):
    aggs.append(a)

# Convert to DataFrame
df = pd.DataFrame(aggs)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Rename columns to match mplfinance requirements
df.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Plot the data
mpf.plot(df, type='candle', style='charles', title='NVDA', volume=True)