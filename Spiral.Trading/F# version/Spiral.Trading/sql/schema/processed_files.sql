-- Tracks which CSV files have been ingested
CREATE TABLE IF NOT EXISTS processed_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL UNIQUE,
    ingested_at TEXT NOT NULL  -- ISO 8601 format
);
