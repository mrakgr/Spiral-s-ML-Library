module Spiral.Trading.Database

open System
open System.Data
open System.IO
open System.Reflection
open Dapper
open Microsoft.Data.Sqlite

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

    connection.Execute(dailyPricesSql) |> ignore
    connection.Execute(splitsSql) |> ignore

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

/// Insert or update multiple daily price records using multi-row inserts
let upsertDailyPrices (sqliteConn : SqliteConnection) (prices: DailyPrice array) : int =
    if prices.Length = 0 then 0
    else
        let batchSize = 100 // 8 params * 100 = 800, under SQLite's 999 limit
        use transaction = sqliteConn.BeginTransaction()
        
        let mutable count = 0
        let mutable i = 0
        
        while i < prices.Length do
            let remaining = prices.Length - i
            let currentBatchSize = min batchSize remaining
            
            use cmd = sqliteConn.CreateCommand()
            cmd.Transaction <- transaction
            cmd.CommandText <- buildMultiRowDailyPriceSql currentBatchSize
            
            for j in 0 .. currentBatchSize - 1 do
                let price = prices.[i + j]
                cmd.Parameters.AddWithValue(sprintf "@t%d" j, price.Ticker) |> ignore
                cmd.Parameters.AddWithValue(sprintf "@d%d" j, price.Date.ToString("yyyy-MM-dd")) |> ignore
                cmd.Parameters.AddWithValue(sprintf "@o%d" j, float price.Open) |> ignore
                cmd.Parameters.AddWithValue(sprintf "@h%d" j, float price.High) |> ignore
                cmd.Parameters.AddWithValue(sprintf "@l%d" j, float price.Low) |> ignore
                cmd.Parameters.AddWithValue(sprintf "@c%d" j, float price.Close) |> ignore
                cmd.Parameters.AddWithValue(sprintf "@v%d" j, price.Volume) |> ignore
                cmd.Parameters.AddWithValue(sprintf "@x%d" j, price.Transactions) |> ignore
            
            count <- count + cmd.ExecuteNonQuery()
            i <- i + currentBatchSize

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
