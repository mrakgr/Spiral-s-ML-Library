"""
Get NVIDIA stock splits from Massive API to verify split adjustment is needed
"""
import json
from massive import RESTClient

# Load API key
with open("trading/api_key.json") as f:
    api_keys = json.load(f)

api_key = api_keys.get("massive_api_key")
client = RESTClient(api_key)

print("Fetching NVIDIA (NVDA) stock splits...")
print("="*60)

# Get splits for NVDA
try:
    splits = client.list_splits(ticker="NVDA", order="asc", limit=1000)
    
    if splits:
        print(f"Found {len(list(splits))} stock splits for NVDA:\n")
        
        # Re-fetch to iterate (generator exhausted)
        splits = client.list_splits(ticker="NVDA", order="asc", limit=1000)
        
        for split in splits:
            print(f"Date: {split.execution_date}")
            print(f"  Ratio: {split.split_from}:{split.split_to} ({split.split_to/split.split_from:.2f}x)")
            print()
    else:
        print("No splits found for NVDA")
        
except Exception as e:
    print(f"Error: {e}")
