#r "nuget: Microsoft.Data.Sqlite, 10.0.1"
#r "nuget: Dapper, 2.1.66"
#r "../bin/Debug/net9.0/Spiral.Trading.dll"

open Dapper

let dbPath = "../../../Python version/data/trading.db"
let conn = Spiral.Trading.Database.openConnection dbPath
Spiral.Trading.Database.initializeSchema conn

printfn "Checking view chain..."

let count (sql: string) = SqlMapper.ExecuteScalar<int64>(conn, sql)

printfn "daily_prices: %d" (count "SELECT COUNT(*) FROM daily_prices")
printfn "split_adjusted_prices: %d" (count "SELECT COUNT(*) FROM split_adjusted_prices")
printfn "trading_calendar: %d" (count "SELECT COUNT(*) FROM trading_calendar")
printfn "stock_momentum_26w: %d" (count "SELECT COUNT(*) FROM stock_momentum_26w")
printfn "stock_dollar_volume_4w: %d" (count "SELECT COUNT(*) FROM stock_dollar_volume_4w")
printfn "stock_momentum_ranking: %d" (count "SELECT COUNT(*) FROM stock_momentum_ranking")
printfn "dom_indicator: %d" (count "SELECT COUNT(*) FROM dom_indicator")

conn.Close()
