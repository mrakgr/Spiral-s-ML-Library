import json
import boto3
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

# Load credentials
with open("api_key.json") as f:
    api_keys = json.load(f)

s3 = boto3.client(
    's3',
    aws_access_key_id=api_keys["massive_s3_access_key"],
    aws_secret_access_key=api_keys["massive_s3_secret_key"],
    endpoint_url='https://files.massive.com'
)

bucket_name = 'flatfiles'

def test_date(date_obj):
    """Test if a specific date is accessible"""
    year = date_obj.year
    month = f"{date_obj.month:02d}"
    date_str = date_obj.strftime("%Y-%m-%d")
    test_key = f"us_stocks_sip/day_aggs_v1/{year}/{month}/{date_str}.csv.gz"
    
    try:
        response = s3.head_object(Bucket=bucket_name, Key=test_key)
        return True, response['ContentLength']
    except ClientError as e:
        return False, e.response['Error']['Code']

print("Finding the exact cutoff date for your subscription...")
print("="*60)

# We know 2003-09 is accessible, let's find the most recent accessible date
# Using binary search between 2003 and now

oldest = datetime(2003, 9, 10)
newest = datetime.now()

# First check if oldest works
works, info = test_date(oldest)
print(f"Oldest (2003-09-10): {'✓ Accessible' if works else f'✗ {info}'}")

# Now find the newest accessible date
print("\nSearching for newest accessible date...")

# Start from recent dates and go backwards in bigger steps
for months_back in [1, 2, 3, 4, 5, 6, 9, 12, 18, 24]:
    test_date_obj = datetime.now() - timedelta(days=months_back*30)
    works, info = test_date(test_date_obj)
    date_str = test_date_obj.strftime("%Y-%m-%d")
    
    if works:
        print(f"✓ {date_str} ({months_back} months ago): Accessible, {info:,} bytes")
        break
    else:
        print(f"✗ {date_str} ({months_back} months ago): {info}")

# If we found a working date, narrow down the exact cutoff
if works:
    print("\nNarrowing down the exact cutoff...")
    
    # Start from that working date and go forward day by day
    current = test_date_obj
    last_working = current
    
    for days_forward in range(0, 60):
        check_date = current + timedelta(days=days_forward)
        if check_date > datetime.now():
            break
            
        works, info = test_date(check_date)
        date_str = check_date.strftime("%Y-%m-%d")
        
        if works:
            last_working = check_date
            print(f"✓ {date_str}: Accessible")
        else:
            if info == '404':
                print(f"  {date_str}: Not found (weekend/holiday)")
            else:
                print(f"✗ {date_str}: {info} - CUTOFF FOUND")
                break
    
    print("\n" + "="*60)
    print(f"RESULT: Most recent accessible date is approximately {last_working.strftime('%Y-%m-%d')}")
    print(f"That's {(datetime.now() - last_working).days} days ago")
    
    # Recommend date range
    recommended_end = last_working - timedelta(days=5)  # Add buffer
    recommended_start = recommended_end - timedelta(days=730)
    
    print("\n" + "="*60)
    print("RECOMMENDED DATE RANGE FOR DOWNLOAD:")
    print(f"  Start: {recommended_start.strftime('%Y-%m-%d')}")
    print(f"  End:   {recommended_end.strftime('%Y-%m-%d')}")
    print(f"  Total: {(recommended_end - recommended_start).days} days")
