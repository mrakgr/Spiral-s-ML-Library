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

print("Finding the most recent available date...")

# Try dates going backwards from today
current_date = datetime.now()
found_date = None

for days_back in range(0, 60):
    test_date = current_date - timedelta(days=days_back)
    year = test_date.year
    month = f"{test_date.month:02d}"
    date_str = test_date.strftime("%Y-%m-%d")
    
    test_key = f"us_stocks_sip/day_aggs_v1/{year}/{month}/{date_str}.csv.gz"
    
    try:
        response = s3.head_object(Bucket=bucket_name, Key=test_key)
        found_date = date_str
        print(f"✓ Found available file: {date_str}")
        print(f"  Size: {response['ContentLength']:,} bytes")
        print(f"  Last Modified: {response['LastModified']}")
        break
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"  {date_str}: Not found (weekend/holiday?)")
        elif e.response['Error']['Code'] == '403':
            print(f"  {date_str}: Access denied (subscription limitation)")
        else:
            print(f"  {date_str}: {e.response['Error']['Code']}")

if found_date:
    print(f"\n✓ Most recent accessible date: {found_date}")
    print(f"\nYour subscription appears to have data up to {found_date}")
else:
    print("\n✗ Could not find any accessible files in the last 60 days")
    print("Your subscription might only include older historical data")

# Check what's the oldest available date too
print("\n" + "="*60)
print("Checking oldest available date by listing...")
try:
    response = s3.list_objects_v2(
        Bucket=bucket_name, 
        Prefix='us_stocks_sip/day_aggs_v1/',
        MaxKeys=1
    )
    if 'Contents' in response:
        oldest_key = response['Contents'][0]['Key']
        print(f"Oldest file: {oldest_key}")
except Exception as e:
    print(f"Error: {e}")
