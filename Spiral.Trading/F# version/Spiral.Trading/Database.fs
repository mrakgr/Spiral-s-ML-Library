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

/// Insert or update multiple daily price records using prepared statement
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
    let sql = "SELECT MIN(date), MAX(date) FROM daily_prices WHERE date IS NOT NULL"
    let result = connection.QueryFirstOrDefault<{| Item1: string; Item2: string |}>(sql)
    if isNull (box result) || String.IsNullOrEmpty(result.Item1) then
        None
    else
        Some (DateTime.Parse(result.Item1), DateTime.Parse(result.Item2))
