module Spiral.Trading.Database

open System
open System.Data
open System.IO
open System.Reflection
open Dapper
open DuckDB.NET.Data

// Row types for Dapper mapping (matches DuckDB column names)
[<CLIMutable>]
type DailyPriceRow = {
    ticker: string
    date: DateOnly
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
    execution_date: DateOnly
    split_from: float
    split_to: float
    split_ratio: float
}

[<CLIMutable>]
type SplitAdjustedPriceRow = {
    ticker: string
    date: DateOnly
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

/// Create and open a DuckDB connection
let openConnection (dbPath: string) : DuckDBConnection =
    let connectionString = $"Data Source={dbPath}"
    let connection = new DuckDBConnection(connectionString)
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

// Note: DuckDB is columnar and optimized for bulk loads by default.
// No PRAGMA statements or index manipulation needed.

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
    VALUES ($ticker, $date, $open, $high, $low, $close, $volume, $transactions)
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

/// Insert or update multiple daily price records (legacy row-by-row method)
let upsertDailyPrices (duckDbConn : DuckDBConnection) (prices: DailyPrice array) : int =
   use transaction = duckDbConn.BeginTransaction()
   use cmd = duckDbConn.CreateCommand()
   cmd.Transaction <- transaction
   cmd.CommandText <- dailyPriceUpsertSql
   let pTicker = new DuckDBParameter("ticker", null)
   let pDate = new DuckDBParameter("date", null)
   let pOpen = new DuckDBParameter("open", null)
   let pHigh = new DuckDBParameter("high", null)
   let pLow = new DuckDBParameter("low", null)
   let pClose = new DuckDBParameter("close", null)
   let pVolume = new DuckDBParameter("volume", null)
   let pTransactions = new DuckDBParameter("transactions", null)
   cmd.Parameters.Add(pTicker) |> ignore
   cmd.Parameters.Add(pDate) |> ignore
   cmd.Parameters.Add(pOpen) |> ignore
   cmd.Parameters.Add(pHigh) |> ignore
   cmd.Parameters.Add(pLow) |> ignore
   cmd.Parameters.Add(pClose) |> ignore
   cmd.Parameters.Add(pVolume) |> ignore
   cmd.Parameters.Add(pTransactions) |> ignore

   let mutable count = 0
   for price in prices do
       pTicker.Value <- price.Ticker
       pDate.Value <- price.Date.ToString("yyyy-MM-dd")
       pOpen.Value <- price.Open
       pHigh.Value <- price.High
       pLow.Value <- price.Low
       pClose.Value <- price.Close
       pVolume.Value <- price.Volume
       pTransactions.Value <- price.Transactions
       count <- count + cmd.ExecuteNonQuery()
   transaction.Commit()
   count

/// Bulk ingest daily prices directly from a .csv.gz file using DuckDB's native CSV reader
let ingestDailyPricesFromCsvGz (connection: IDbConnection) (filePath: string) : int64 =
    let sql = $"""
        INSERT INTO daily_prices (ticker, date, open, high, low, close, volume, transactions)
        SELECT 
            ticker,
            (epoch_ms(0) + to_milliseconds(window_start / 1000000))::DATE as date,
            open, high, low, close, volume, transactions
        FROM read_csv('{filePath}',
            columns = {{
                'ticker': 'VARCHAR',
                'volume': 'BIGINT',
                'open': 'DOUBLE',
                'close': 'DOUBLE',
                'high': 'DOUBLE',
                'low': 'DOUBLE',
                'window_start': 'BIGINT',
                'transactions': 'BIGINT'
            }},
            header = true
        )
        ON CONFLICT(ticker, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            transactions = excluded.transactions
    """
    connection.Execute(sql) |> int64

/// Bulk ingest daily prices from multiple .csv.gz files at once using glob pattern
let ingestDailyPricesFromGlob (connection: IDbConnection) (globPattern: string) : int64 =
    let sql = $"""
        INSERT INTO daily_prices (ticker, date, open, high, low, close, volume, transactions)
        SELECT 
            ticker,
            (epoch_ms(0) + to_milliseconds(window_start / 1000000))::DATE as date,
            open, high, low, close, volume, transactions
        FROM read_csv('{globPattern}',
            columns = {{
                'ticker': 'VARCHAR',
                'volume': 'BIGINT',
                'open': 'DOUBLE',
                'close': 'DOUBLE',
                'high': 'DOUBLE',
                'low': 'DOUBLE',
                'window_start': 'BIGINT',
                'transactions': 'BIGINT'
            }},
            header = true
        )
        ON CONFLICT(ticker, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            transactions = excluded.transactions
    """
    connection.Execute(sql) |> int64

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
    VALUES ($ticker, $execution_date, $split_from, $split_to, $split_ratio)
    ON CONFLICT(ticker, execution_date) DO UPDATE SET
        split_from = excluded.split_from,
        split_to = excluded.split_to,
        split_ratio = excluded.split_ratio
"""

/// Insert or update a single split record
let upsertSplit (connection: IDbConnection) (split: Split) : int =
    connection.Execute(splitUpsertSql, toSplitParams split)

/// Insert or update multiple split records using prepared statement (legacy row-by-row method)
let upsertSplits (connection: IDbConnection) (splits: Split array) : int =
    let duckDbConn = connection :?> DuckDBConnection
    use transaction = duckDbConn.BeginTransaction()
    use cmd = duckDbConn.CreateCommand()
    cmd.Transaction <- transaction
    cmd.CommandText <- splitUpsertSql

    let pTicker = new DuckDBParameter("ticker", null)
    let pExecutionDate = new DuckDBParameter("execution_date", null)
    let pSplitFrom = new DuckDBParameter("split_from", null)
    let pSplitTo = new DuckDBParameter("split_to", null)
    let pSplitRatio = new DuckDBParameter("split_ratio", null)
    cmd.Parameters.Add(pTicker) |> ignore
    cmd.Parameters.Add(pExecutionDate) |> ignore
    cmd.Parameters.Add(pSplitFrom) |> ignore
    cmd.Parameters.Add(pSplitTo) |> ignore
    cmd.Parameters.Add(pSplitRatio) |> ignore

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

/// Bulk ingest splits directly from a CSV file using DuckDB's native CSV reader
let ingestSplitsFromCsv (connection: IDbConnection) (filePath: string) : int64 =
    let sql = $"""
        INSERT INTO splits (ticker, execution_date, split_from, split_to, split_ratio)
        SELECT 
            ticker,
            execution_date::DATE,
            split_from,
            split_to,
            split_ratio
        FROM read_csv('{filePath}',
            columns = {{
                'ticker': 'VARCHAR',
                'execution_date': 'VARCHAR',
                'split_from': 'DOUBLE',
                'split_to': 'DOUBLE',
                'split_ratio': 'DOUBLE'
            }},
            header = true
        )
        ON CONFLICT(ticker, execution_date) DO UPDATE SET
            split_from = excluded.split_from,
            split_to = excluded.split_to,
            split_ratio = excluded.split_ratio
    """
    connection.Execute(sql) |> int64

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
    let minDate = connection.ExecuteScalar<obj>("SELECT MIN(date) FROM daily_prices WHERE date IS NOT NULL")
    let maxDate = connection.ExecuteScalar<obj>("SELECT MAX(date) FROM daily_prices WHERE date IS NOT NULL")
    if isNull minDate || isNull maxDate then
        None
    else
        let toDateTime (o: obj) =
            match o with
            | :? DateOnly as d -> d.ToDateTime(TimeOnly.MinValue)
            | :? DateTime as d -> d
            | _ -> Convert.ToDateTime(o)
        Some (toDateTime minDate, toDateTime maxDate)

/// Get daily prices for a specific ticker, ordered by date
let getDailyPricesByTicker (connection: IDbConnection) (ticker: string) : DailyPrice array =
    connection.Query<DailyPriceRow>(
        "SELECT ticker, date, open, high, low, close, volume, transactions FROM daily_prices WHERE ticker = $ticker ORDER BY date",
        {| ticker = ticker |})
    |> Seq.map (fun row -> {
        Ticker = row.ticker
        Date = row.date.ToDateTime(TimeOnly.MinValue)
        Open = row.``open``
        High = row.high
        Low = row.low
        Close = row.close
        Volume = row.volume
        Transactions = row.transactions
    })
    |> Seq.toArray

/// Get splits for a specific ticker, ordered by execution date descending
let getSplitsByTicker (connection: IDbConnection) (ticker: string) : Split array =
    connection.Query<SplitRow>(
        "SELECT ticker, execution_date, split_from, split_to, split_ratio FROM splits WHERE ticker = $ticker ORDER BY execution_date DESC",
        {| ticker = ticker |})
    |> Seq.map (fun row -> {
        Ticker = row.ticker
        ExecutionDate = row.execution_date.ToDateTime(TimeOnly.MinValue)
        SplitFrom = row.split_from
        SplitTo = row.split_to
        SplitRatio = row.split_ratio
    })
    |> Seq.toArray

/// Get split-adjusted daily prices for a specific ticker using the split_adjusted_prices view
let getSplitAdjustedPricesByTicker (connection: IDbConnection) (ticker: string) : SplitAdjustedPriceRow array =
    connection.Query<SplitAdjustedPriceRow>(
        "SELECT ticker, date, adj_open, adj_high, adj_low, adj_close, adj_volume FROM split_adjusted_prices WHERE ticker = $ticker ORDER BY date",
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
        "INSERT INTO processed_files (file_name, ingested_at) VALUES ($fileName, $ingestedAt) ON CONFLICT DO NOTHING",
        {| fileName = fileName; ingestedAt = DateTime.UtcNow.ToString("o") |}) |> ignore

/// Mark multiple files as processed
let markFilesProcessed (connection: IDbConnection) (fileNames: string seq) : unit =
    let duckDbConn = connection :?> DuckDBConnection
    use transaction = duckDbConn.BeginTransaction()
    use cmd = duckDbConn.CreateCommand()
    cmd.Transaction <- transaction
    cmd.CommandText <- "INSERT INTO processed_files (file_name, ingested_at) VALUES ($fileName, $ingestedAt) ON CONFLICT DO NOTHING"
    let pFileName = new DuckDBParameter("fileName", null)
    let pIngestedAt = new DuckDBParameter("ingestedAt", null)
    cmd.Parameters.Add(pFileName) |> ignore
    cmd.Parameters.Add(pIngestedAt) |> ignore
    let now = DateTime.UtcNow.ToString("o")

    for fileName in fileNames do
        pFileName.Value <- fileName
        pIngestedAt.Value <- now
        cmd.ExecuteNonQuery() |> ignore

    transaction.Commit()

/// Bulk ingest daily prices from multiple .csv.gz files with progress reporting
let ingestDailyPricesFromDirectory 
    (connection: IDbConnection) 
    (csvDir: string) 
    (progress: int -> int -> string -> int64 -> unit) 
    : int64 =
    let processedFiles = getProcessedFiles connection
    let allFiles = Directory.GetFiles(csvDir, "*.csv.gz")
    let filesToProcess = 
        allFiles 
        |> Array.filter (fun f -> not (processedFiles.Contains(Path.GetFileName(f))))
    
    let mutable totalRows = 0L
    let mutable filesProcessed = 0
    let totalFiles = filesToProcess.Length
    
    for filePath in filesToProcess do
        let fileName = Path.GetFileName(filePath)
        let rowsInserted = ingestDailyPricesFromCsvGz connection filePath
        markFileProcessed connection fileName
        filesProcessed <- filesProcessed + 1
        totalRows <- totalRows + rowsInserted
        progress filesProcessed totalFiles fileName rowsInserted
    
    totalRows

// --- DOM Indicator ---

[<CLIMutable>]
type DomIndicatorRow = {
    date: DateOnly
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
