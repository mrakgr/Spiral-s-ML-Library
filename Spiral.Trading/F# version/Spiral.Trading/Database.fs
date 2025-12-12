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

/// Load embedded SQL resource by name
let private loadEmbeddedSql (resourceName: string) : string =
    let assembly = Assembly.GetExecutingAssembly()
    let fullName =
        assembly.GetManifestResourceNames()
        |> Array.find (fun n -> n.EndsWith(resourceName))

    use stream = assembly.GetManifestResourceStream(fullName)
    use reader = new StreamReader(stream)
    reader.ReadToEnd()

/// Create and open a SQLite connection
let openConnection (dbPath: string) : SqliteConnection =
    let connectionString = $"Data Source={dbPath}"
    let connection = new SqliteConnection(connectionString)
    connection.Open()
    connection

/// Initialize the database schema
let initializeSchema (connection: IDbConnection) : unit =
    let dailyPricesSql = loadEmbeddedSql "daily_prices.sql"
    let splitsSql = loadEmbeddedSql "splits.sql"
    let processedFilesSql = loadEmbeddedSql "processed_files.sql"

    connection.Execute(dailyPricesSql) |> ignore
    connection.Execute(splitsSql) |> ignore
    connection.Execute(processedFilesSql) |> ignore

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

/// Build multi-row INSERT SQL for a batch of daily prices
let private buildMultiRowDailyPriceSql (batchSize: int) : string =
    let values = 
        [| for i in 0 .. batchSize - 1 ->
            sprintf "(@t%d, @d%d, @o%d, @h%d, @l%d, @c%d, @v%d, @x%d)" i i i i i i i i |]
        |> String.concat ", "
    sprintf """
        INSERT INTO daily_prices (ticker, date, open, high, low, close, volume, transactions)
        VALUES %s
        ON CONFLICT(ticker, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            transactions = excluded.transactions
    """ values

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
