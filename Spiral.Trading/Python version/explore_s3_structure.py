import json
import boto3
from pathlib import Path

# Load API credentials
with open("api_key.json") as f:
    api_keys = json.load(f)

s3_access_key = api_keys.get("massive_s3_access_key")
s3_secret_key = api_keys.get("massive_s3_secret_key")

# Configure S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
    endpoint_url='https://files.massive.com'
)

bucket_name = 'flatfiles'

print("Exploring S3 bucket structure...")
print("=" * 60)

# List top-level prefixes
response = s3.list_objects_v2(
    Bucket=bucket_name,
    Delimiter='/',
    MaxKeys=100
)

print("\nTop-level directories:")
if 'CommonPrefixes' in response:
    for prefix in response['CommonPrefixes']:
        print(f"  - {prefix['Prefix']}")

# Explore us_stocks_sip structure
print("\n\nContents of us_stocks_sip/:")
print("-" * 60)
response = s3.list_objects_v2(
    Bucket=bucket_name,
    Prefix='us_stocks_sip/',
    Delimiter='/',
    MaxKeys=100
)

if 'CommonPrefixes' in response:
    for prefix in response['CommonPrefixes']:
        print(f"  - {prefix['Prefix']}")

print("\n\nDone!")
