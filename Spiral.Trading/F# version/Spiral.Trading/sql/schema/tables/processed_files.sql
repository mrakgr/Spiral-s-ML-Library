-- Tracks which CSV files have been ingested
CREATE TABLE IF NOT EXISTS processed_files (
    file_name VARCHAR NOT NULL PRIMARY KEY,
    ingested_at VARCHAR NOT NULL  -- ISO 8601 format
);
