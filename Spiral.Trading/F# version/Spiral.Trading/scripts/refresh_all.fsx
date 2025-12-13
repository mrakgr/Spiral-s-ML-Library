#r "nuget: Microsoft.Data.Sqlite, 10.0.1"
#r "nuget: Dapper, 2.1.66"
#r "../bin/Debug/net9.0/Spiral.Trading.dll"

open System.Diagnostics

let dbPath = "../../../Python version/data/trading.db"
let conn = Spiral.Trading.Database.openConnection dbPath
Spiral.Trading.Database.initializeSchema conn

let sw = Stopwatch.StartNew()
Spiral.Trading.Database.refreshMaterializedTables conn
sw.Stop()

printfn "Total time: %.1f minutes" sw.Elapsed.TotalMinutes

conn.Close()
