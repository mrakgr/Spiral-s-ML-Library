import json
import boto3
from botocore.exceptions import ClientError

# Load credentials
with open("api_key.json") as f:
    api_keys = json.load(f)

s3_access_key = api_keys.get("massive_s3_access_key")
s3_secret_key = api_keys.get("massive_s3_secret_key")

print("Testing S3 access to Massive flat files...")
print(f"Access Key: {s3_access_key[:8]}...")
print(f"Secret Key: {s3_secret_key[:8]}...")

# Configure S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
    endpoint_url='https://files.massive.com'
)

bucket_name = 'flatfiles'

# Test 1: List bucket contents
print("\nTest 1: Listing bucket root...")
try:
    response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
    if 'Contents' in response:
        print("✓ Success! Found files:")
        for obj in response['Contents'][:5]:
            print(f"  - {obj['Key']}")
    else:
        print("✓ Connected but no files found (empty response)")
except ClientError as e:
    error_code = e.response['Error']['Code']
    print(f"✗ Error: {error_code} - {e.response['Error']['Message']}")
    if error_code == '403':
        print("\nPossible causes:")
        print("  1. Your subscription doesn't include Flat Files access")
        print("  2. S3 credentials are incorrect")
        print("  3. Credentials haven't been activated yet")
        print("\nCheck: https://polygon.io/dashboard/flat-files")

# Test 2: Try listing a specific prefix
print("\nTest 2: Listing us_stocks_sip directory...")
try:
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix='us_stocks_sip/', MaxKeys=10)
    if 'Contents' in response:
        print("✓ Success! Found files:")
        for obj in response['Contents'][:5]:
            print(f"  - {obj['Key']}")
    else:
        print("! No files found in us_stocks_sip/")
except ClientError as e:
    print(f"✗ Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}")

# Test 3: Try a recent file
print("\nTest 3: Checking for a recent daily aggregate file...")
try:
    from datetime import datetime, timedelta
    recent_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    year = recent_date[:4]
    month = recent_date[5:7]
    test_key = f"us_stocks_sip/day_aggs_v1/{year}/{month}/{recent_date}.csv.gz"
    
    print(f"Trying: {test_key}")
    response = s3.head_object(Bucket=bucket_name, Key=test_key)
    print(f"✓ File exists! Size: {response['ContentLength']:,} bytes")
except ClientError as e:
    print(f"✗ Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}")

print("\n" + "="*60)
print("Diagnosis complete.")
