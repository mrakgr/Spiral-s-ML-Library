module Spiral.Trading.Database

open System
open System.Data
open System.IO
open System.Reflection
open Dapper
open Microsoft.Data.Sqlite

// Row types for Dapper mapping (matches SQLite column names)
[<CLIMutable>]
type DailyPriceRow = {
    ticker: string
    date: string
    ``open``: float
    high: float
    low: float
    close: float
    volume: int64
    transactions: int64
}

[<CLIMutable>]
type SplitRow = {
    ticker: string
    execution_date: string
    split_from: float
    split_to: float
    split_ratio: float
}

[<CLIMutable>]
type SplitAdjustedPriceRow = {
    ticker: string
    date: string
    adj_open: float
    adj_high: float
    adj_low: float
    adj_close: float
    adj_volume: int64
}

/// Load embedded SQL resource by name
let private loadEmbeddedSql (resourceName: string) : string =
    let assembly = Assembly.GetExecutingAssembly()
    let fullName =
        assembly.GetManifestResourceNames()
        |> Array.find (fun n -> n.EndsWith(resourceName))

    use stream = assembly.GetManifestResourceStream(fullName)
    use reader = new StreamReader(stream)
    reader.ReadToEnd()

/// Get all embedded SQL resources from a specific folder
let private getEmbeddedSqlFromFolder (folderName: string) : string array =
    let assembly = Assembly.GetExecutingAssembly()
    assembly.GetManifestResourceNames()
    |> Array.filter (fun n -> n.Contains(folderName) && n.EndsWith(".sql"))
    |> Array.sort

/// Create and open a SQLite connection
let openConnection (dbPath: string) : SqliteConnection =
    let connectionString = $"Data Source={dbPath}"
    let connection = new SqliteConnection(connectionString)
    connection.Open()
    connection

/// Initialize the database schema
let initializeSchema (connection: IDbConnection) : unit =
    let assembly = Assembly.GetExecutingAssembly()

    // Executes the sql.
    let executeSql folderName =
        for resourceName in getEmbeddedSqlFromFolder folderName do
            use stream = assembly.GetManifestResourceStream(resourceName)
            use reader = new StreamReader(stream)
            let sql = reader.ReadToEnd()
            connection.Execute(sql) |> ignore
    
    // Execute all table schemas first
    executeSql "sql.schema.tables"
    
    // Execute all view schemas
    executeSql "sql.schema.views"

/// Apply PRAGMA optimizations for bulk loading
let applyBulkLoadPragmas (connection: IDbConnection) : unit =
    connection.Execute("PRAGMA synchronous = OFF") |> ignore
    connection.Execute("PRAGMA journal_mode = MEMORY") |> ignore
    connection.Execute("PRAGMA temp_store = MEMORY") |> ignore
    connection.Execute("PRAGMA cache_size = -64000") |> ignore // 64MB cache

/// Restore default PRAGMA settings after bulk loading
let restoreDefaultPragmas (connection: IDbConnection) : unit =
    connection.Execute("PRAGMA synchronous = FULL") |> ignore
    connection.Execute("PRAGMA journal_mode = DELETE") |> ignore
    connection.Execute("PRAGMA temp_store = DEFAULT") |> ignore
    connection.Execute("PRAGMA cache_size = -2000") |> ignore // Default ~2MB

/// Drop all indexes for bulk loading performance
let dropIndexes (connection: IDbConnection) : unit =
    connection.Execute("DROP INDEX IF EXISTS idx_daily_prices_ticker") |> ignore
    connection.Execute("DROP INDEX IF EXISTS idx_daily_prices_date") |> ignore
    connection.Execute("DROP INDEX IF EXISTS idx_daily_prices_ticker_date") |> ignore
    connection.Execute("DROP INDEX IF EXISTS idx_splits_ticker") |> ignore
    connection.Execute("DROP INDEX IF EXISTS idx_splits_execution_date") |> ignore

/// Recreate all indexes after bulk loading
let recreateIndexes (connection: IDbConnection) : unit =
    connection.Execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices(ticker)") |> ignore
    connection.Execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date)") |> ignore
    connection.Execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date ON daily_prices(ticker, date)") |> ignore
    connection.Execute("CREATE INDEX IF NOT EXISTS idx_splits_ticker ON splits(ticker)") |> ignore
    connection.Execute("CREATE INDEX IF NOT EXISTS idx_splits_execution_date ON splits(execution_date)") |> ignore

/// Execute a bulk load operation with optimized settings
let withBulkLoadOptimizations (connection: IDbConnection) (operation: unit -> 'a) : 'a =
    applyBulkLoadPragmas connection
    dropIndexes connection
    try
        let result = operation ()
        recreateIndexes connection
        restoreDefaultPragmas connection
        result
    with ex ->
        recreateIndexes connection
        restoreDefaultPragmas connection
        reraise ()

/// Convert DailyPrice to Dapper DynamicParameters
let private toDailyPriceParams (price: DailyPrice) : DynamicParameters =
    let p = DynamicParameters()
    p.Add("ticker", price.Ticker)
    p.Add("date", price.Date.ToString("yyyy-MM-dd"))
    p.Add("open", price.Open)
    p.Add("high", price.High)
    p.Add("low", price.Low)
    p.Add("close", price.Close)
    p.Add("volume", price.Volume)
    p.Add("transactions", price.Transactions)
    p

let private dailyPriceUpsertSql = """
    INSERT INTO daily_prices (ticker, date, open, high, low, close, volume, transactions)
    VALUES (@ticker, @date, @open, @high, @low, @close, @volume, @transactions)
    ON CONFLICT(ticker, date) DO UPDATE SET
        open = excluded.open,
        high = excluded.high,
        low = excluded.low,
        close = excluded.close,
        volume = excluded.volume,
        transactions = excluded.transactions
"""

/// Insert or update a single daily price record
let upsertDailyPrice (connection: IDbConnection) (price: DailyPrice) : int =
    connection.Execute(dailyPriceUpsertSql, toDailyPriceParams price)

/// Insert or update multiple daily price records
let upsertDailyPrices (sqliteConn : SqliteConnection) (prices: DailyPrice array) : int =
   use transaction = sqliteConn.BeginTransaction()
   use cmd = sqliteConn.CreateCommand()
   cmd.Transaction <- transaction
   cmd.CommandText <- dailyPriceUpsertSql
   let pTicker = cmd.Parameters.Add("@ticker", SqliteType.Text)
   let pDate = cmd.Parameters.Add("@date", SqliteType.Text)
   let pOpen = cmd.Parameters.Add("@open", SqliteType.Real)
   let pHigh = cmd.Parameters.Add("@high", SqliteType.Real)
   let pLow = cmd.Parameters.Add("@low", SqliteType.Real)
   let pClose = cmd.Parameters.Add("@close", SqliteType.Real)
   let pVolume = cmd.Parameters.Add("@volume", SqliteType.Integer)
   let pTransactions = cmd.Parameters.Add("@transactions", SqliteType.Integer)
   
   let mutable count = 0
   for price in prices do
       pTicker.Value <- price.Ticker
       pDate.Value <- price.Date.ToString("yyyy-MM-dd")
       pOpen.Value <- float price.Open
       pHigh.Value <- float price.High
       pLow.Value <- float price.Low
       pClose.Value <- float price.Close
       pVolume.Value <- price.Volume
       pTransactions.Value <- price.Transactions
       count <- count + cmd.ExecuteNonQuery()
   transaction.Commit()
   count  

/// Convert Split to Dapper DynamicParameters
let private toSplitParams (split: Split) : DynamicParameters =
    let p = DynamicParameters()
    p.Add("ticker", split.Ticker)
    p.Add("execution_date", split.ExecutionDate.ToString("yyyy-MM-dd"))
    p.Add("split_from", split.SplitFrom)
    p.Add("split_to", split.SplitTo)
    p.Add("split_ratio", split.SplitRatio)
    p

let private splitUpsertSql = """
    INSERT INTO splits (ticker, execution_date, split_from, split_to, split_ratio)
    VALUES (@ticker, @execution_date, @split_from, @split_to, @split_ratio)
    ON CONFLICT(ticker, execution_date) DO UPDATE SET
        split_from = excluded.split_from,
        split_to = excluded.split_to,
        split_ratio = excluded.split_ratio
"""

/// Insert or update a single split record
let upsertSplit (connection: IDbConnection) (split: Split) : int =
    connection.Execute(splitUpsertSql, toSplitParams split)

/// Insert or update multiple split records using prepared statement
let upsertSplits (connection: IDbConnection) (splits: Split array) : int =
    let sqliteConn = connection :?> SqliteConnection
    use transaction = sqliteConn.BeginTransaction()
    use cmd = sqliteConn.CreateCommand()
    cmd.Transaction <- transaction
    cmd.CommandText <- splitUpsertSql

    let pTicker = cmd.Parameters.Add("@ticker", SqliteType.Text)
    let pExecutionDate = cmd.Parameters.Add("@execution_date", SqliteType.Text)
    let pSplitFrom = cmd.Parameters.Add("@split_from", SqliteType.Real)
    let pSplitTo = cmd.Parameters.Add("@split_to", SqliteType.Real)
    let pSplitRatio = cmd.Parameters.Add("@split_ratio", SqliteType.Real)

    let mutable count = 0
    for split in splits do
        pTicker.Value <- split.Ticker
        pExecutionDate.Value <- split.ExecutionDate.ToString("yyyy-MM-dd")
        pSplitFrom.Value <- split.SplitFrom
        pSplitTo.Value <- split.SplitTo
        pSplitRatio.Value <- split.SplitRatio
        count <- count + cmd.ExecuteNonQuery()

    transaction.Commit()
    count

/// Get count of daily prices in database
let getDailyPriceCount (connection: IDbConnection) : int64 =
    connection.ExecuteScalar<int64>("SELECT COUNT(*) FROM daily_prices")

/// Get count of splits in database
let getSplitCount (connection: IDbConnection) : int64 =
    connection.ExecuteScalar<int64>("SELECT COUNT(*) FROM splits")

/// Get all unique tickers from daily prices
let getTickers (connection: IDbConnection) : string array =
    connection.Query<string>("SELECT DISTINCT ticker FROM daily_prices ORDER BY ticker")
    |> Seq.toArray

/// Get date range for daily prices
let getDateRange (connection: IDbConnection) : (DateTime * DateTime) option =
    let minDate = connection.ExecuteScalar<string>("SELECT MIN(date) FROM daily_prices WHERE date IS NOT NULL")
    let maxDate = connection.ExecuteScalar<string>("SELECT MAX(date) FROM daily_prices WHERE date IS NOT NULL")
    if String.IsNullOrEmpty(minDate) || String.IsNullOrEmpty(maxDate) then
        None
    else
        Some (DateTime.Parse(minDate), DateTime.Parse(maxDate))

/// Get daily prices for a specific ticker, ordered by date
let getDailyPricesByTicker (connection: IDbConnection) (ticker: string) : DailyPrice array =
    connection.Query<DailyPriceRow>(
        "SELECT ticker, date, open, high, low, close, volume, transactions FROM daily_prices WHERE ticker = @ticker ORDER BY date",
        {| ticker = ticker |})
    |> Seq.map (fun row -> {
        Ticker = row.ticker
        Date = DateTime.Parse(row.date)
        Open = decimal row.``open``
        High = decimal row.high
        Low = decimal row.low
        Close = decimal row.close
        Volume = row.volume
        Transactions = row.transactions
    })
    |> Seq.toArray

/// Get splits for a specific ticker, ordered by execution date descending
let getSplitsByTicker (connection: IDbConnection) (ticker: string) : Split array =
    connection.Query<SplitRow>(
        "SELECT ticker, execution_date, split_from, split_to, split_ratio FROM splits WHERE ticker = @ticker ORDER BY execution_date DESC",
        {| ticker = ticker |})
    |> Seq.map (fun row -> {
        Ticker = row.ticker
        ExecutionDate = DateTime.Parse(row.execution_date)
        SplitFrom = row.split_from
        SplitTo = row.split_to
        SplitRatio = row.split_ratio
    })
    |> Seq.toArray

/// Get split-adjusted daily prices for a specific ticker using the split_adjusted_prices view
let getSplitAdjustedPricesByTicker (connection: IDbConnection) (ticker: string) : SplitAdjustedPriceRow array =
    connection.Query<SplitAdjustedPriceRow>(
        "SELECT ticker, date, adj_open, adj_high, adj_low, adj_close, adj_volume FROM split_adjusted_prices WHERE ticker = @ticker ORDER BY date",
        {| ticker = ticker |})
    |> Seq.toArray

// --- Processed Files Tracking ---

/// Get set of already processed file names
let getProcessedFiles (connection: IDbConnection) : Set<string> =
    connection.Query<string>("SELECT file_name FROM processed_files")
    |> Set.ofSeq

/// Mark a file as processed
let markFileProcessed (connection: IDbConnection) (fileName: string) : unit =
    connection.Execute(
        "INSERT OR IGNORE INTO processed_files (file_name, ingested_at) VALUES (@fileName, @ingestedAt)",
        {| fileName = fileName; ingestedAt = DateTime.UtcNow.ToString("o") |}) |> ignore

/// Mark multiple files as processed
let markFilesProcessed (connection: IDbConnection) (fileNames: string seq) : unit =
    let sqliteConn = connection :?> SqliteConnection
    use transaction = sqliteConn.BeginTransaction()
    use cmd = sqliteConn.CreateCommand()
    cmd.Transaction <- transaction
    cmd.CommandText <- "INSERT OR IGNORE INTO processed_files (file_name, ingested_at) VALUES (@fileName, @ingestedAt)"
    let pFileName = cmd.Parameters.Add("@fileName", SqliteType.Text)
    let pIngestedAt = cmd.Parameters.Add("@ingestedAt", SqliteType.Text)
    let now = DateTime.UtcNow.ToString("o")
    
    for fileName in fileNames do
        pFileName.Value <- fileName
        pIngestedAt.Value <- now
        cmd.ExecuteNonQuery() |> ignore
    
    transaction.Commit()

// --- DOM Indicator ---

[<CLIMutable>]
type DomIndicatorRow = {
    date: string
    avg_leader_return: float
    avg_laggard_return: float
    n_leaders: int64
    n_laggards: int64
    dom_contribution: float
}

/// Get DOM indicator data
let getDomIndicator (connection: IDbConnection) : DomIndicatorRow array =
    connection.Query<DomIndicatorRow>("SELECT * FROM dom_indicator ORDER BY date")
    |> Seq.toArray

// --- Materialized Table Refresh ---

let private refreshStockDollarVolume4wSql = """
    DELETE FROM stock_dollar_volume_4w;
    INSERT INTO stock_dollar_volume_4w (ticker, date, total_dollar_volume, trading_days, avg_dollar_volume_4w)
    SELECT 
        ticker,
        date,
        total_dollar_volume,
        trading_days,
        total_dollar_volume / trading_days AS avg_dollar_volume_4w
    FROM (
        SELECT 
            p.ticker,
            tc.current_date AS date,
            (
                SELECT SUM(p2.adj_close * p2.adj_volume)
                FROM split_adjusted_prices p2
                WHERE p2.ticker = p.ticker
                AND p2.date >= tc.date_4w_ago
                AND p2.date <= tc.current_date
            ) AS total_dollar_volume,
            (
                SELECT COUNT(*)
                FROM split_adjusted_prices p2
                WHERE p2.ticker = p.ticker
                AND p2.date >= tc.date_4w_ago
                AND p2.date <= tc.current_date
            ) AS trading_days
        FROM split_adjusted_prices p
        JOIN trading_calendar tc ON p.date = tc.current_date
    );
"""

let private refreshStockMomentumRankingSql = """
    DELETE FROM stock_momentum_ranking;
    INSERT INTO stock_momentum_ranking (ticker, date, adj_close, momentum_26w, avg_dollar_volume_4w, momentum_rank, total_stocks)
    SELECT 
        ticker,
        date,
        adj_close,
        momentum_26w,
        avg_dollar_volume_4w,
        RANK() OVER (PARTITION BY date ORDER BY momentum_26w DESC) AS momentum_rank,
        COUNT(*) OVER (PARTITION BY date) AS total_stocks
    FROM (
        SELECT 
            m.ticker,
            m.date,
            m.adj_close,
            m.momentum_26w,
            v.avg_dollar_volume_4w
        FROM stock_momentum_26w m
        JOIN stock_dollar_volume_4w v 
            ON v.ticker = m.ticker 
            AND v.date = m.date
        WHERE v.avg_dollar_volume_4w >= 100000000
    );
"""

/// Refresh the stock_dollar_volume_4w materialized table
let refreshStockDollarVolume4w (connection: IDbConnection) : unit =
    printfn "Refreshing stock_dollar_volume_4w..."
    connection.Execute(refreshStockDollarVolume4wSql) |> ignore
    let count = connection.ExecuteScalar<int64>("SELECT COUNT(*) FROM stock_dollar_volume_4w")
    printfn "  Inserted %d rows" count

/// Refresh the stock_momentum_ranking materialized table
let refreshStockMomentumRanking (connection: IDbConnection) : unit =
    printfn "Refreshing stock_momentum_ranking..."
    connection.Execute(refreshStockMomentumRankingSql) |> ignore
    let count = connection.ExecuteScalar<int64>("SELECT COUNT(*) FROM stock_momentum_ranking")
    printfn "  Inserted %d rows" count

/// Refresh all materialized tables in the correct order
let refreshMaterializedTables (connection: IDbConnection) : unit =
    printfn "Refreshing materialized tables..."
    refreshStockDollarVolume4w connection
    refreshStockMomentumRanking connection
    printfn "Done."
