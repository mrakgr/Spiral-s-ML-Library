-- Tracks which CSV files have been ingested
CREATE TABLE IF NOT EXISTS processed_files (
    id INTEGER PRIMARY KEY,
    file_name VARCHAR NOT NULL UNIQUE,
    ingested_at VARCHAR NOT NULL  -- ISO 8601 format
);
