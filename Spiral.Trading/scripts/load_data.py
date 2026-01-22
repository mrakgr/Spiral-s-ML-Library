#!/usr/bin/env python3
"""Load and inspect the training dataset."""

import pyarrow.parquet as pq

def main():
    pf = pq.ParquetFile('data/train.parquet')
    
    print("=== Dataset Info ===")
    print(f"Num row groups: {pf.metadata.num_row_groups}")
    print(f"Num rows: {pf.metadata.num_rows}")
    print(f"Schema: {pf.schema_arrow}")
    print()
    
    print("=== First 100 rows (from first row group) ===")
    df = pf.read_row_group(0).to_pandas()
    print(df.head(100))

if __name__ == "__main__":
    main()
