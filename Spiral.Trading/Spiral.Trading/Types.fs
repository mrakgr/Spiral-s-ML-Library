namespace Spiral.Trading

open System

/// Configuration for Massive API access
type MassiveConfig = {
    ApiKey: string
    S3AccessKey: string
    S3SecretKey: string
}

/// Stock split information from Massive API
type Split = {
    Ticker: string
    ExecutionDate: DateTime
    SplitFrom: float
    SplitTo: float
    SplitRatio: float
}

/// Daily OHLCV price data
type DailyPrice = {
    Ticker: string
    Date: DateTime
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int64
    Transactions: int64
}

/// Result of a download operation for a single date
type DownloadResult =
    | Downloaded of date: DateTime
    | Skipped of date: DateTime
    | Failed of date: DateTime * error: string
