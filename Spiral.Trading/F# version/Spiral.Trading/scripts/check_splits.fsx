#r "nuget: Microsoft.Data.Sqlite, 10.0.1"
#r "nuget: Dapper, 2.1.66"
open Dapper
let conn = new Microsoft.Data.Sqlite.SqliteConnection("Data Source=../../../Python version/data/trading.db")
conn.Open()
let splitCount = conn.ExecuteScalar<int64>("SELECT COUNT(*) FROM splits")
let tickerCount = conn.ExecuteScalar<int64>("SELECT COUNT(DISTINCT ticker) FROM splits")
printfn "Total splits: %d" splitCount
printfn "Tickers with splits: %d" tickerCount
conn.Close()
