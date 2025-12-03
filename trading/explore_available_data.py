import json
import boto3
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

print("Exploring what data types you have access to...")
print("="*60)

# Check different data types
prefixes_to_check = [
    'us_stocks_sip/day_aggs_v1/',
    'us_stocks_sip/minute_aggs_v1/',
    'us_stocks_sip/trades_v1/',
    'us_stocks_sip/quotes_v1/',
    'us_stocks_sip/',
]

for prefix in prefixes_to_check:
    print(f"\nChecking: {prefix}")
    try:
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=5
        )
        
        if 'Contents' in response and len(response['Contents']) > 0:
            print(f"  ✓ Accessible! Found {len(response['Contents'])} files:")
            for obj in response['Contents'][:3]:
                print(f"    - {obj['Key']}")
                
            # Try to actually access one of these files
            test_key = response['Contents'][0]['Key']
            try:
                head_response = s3.head_object(Bucket=bucket_name, Key=test_key)
                print(f"  ✓ Can HEAD this file: {head_response['ContentLength']:,} bytes")
            except ClientError as e:
                print(f"  ✗ Cannot HEAD this file: {e.response['Error']['Code']}")
        else:
            print(f"  ! No contents found (might not exist)")
            
    except ClientError as e:
        print(f"  ✗ Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}")

# List all top-level directories
print("\n" + "="*60)
print("Top-level directories in bucket:")
try:
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Delimiter='/',
        MaxKeys=100
    )
    
    if 'CommonPrefixes' in response:
        for prefix in response['CommonPrefixes']:
            print(f"  - {prefix['Prefix']}")
    else:
        print("  No common prefixes found")
        
except Exception as e:
    print(f"Error listing: {e}")

# Check subscription tier info by looking at what we can actually list
print("\n" + "="*60)
print("Attempting to list all us_stocks_sip subdirectories...")
try:
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix='us_stocks_sip/',
        Delimiter='/',
        MaxKeys=100
    )
    
    if 'CommonPrefixes' in response:
        print("Available data types:")
        for prefix in response['CommonPrefixes']:
            print(f"  - {prefix['Prefix']}")
    else:
        print("  No subdirectories found")
        
except Exception as e:
    print(f"Error: {e}")
