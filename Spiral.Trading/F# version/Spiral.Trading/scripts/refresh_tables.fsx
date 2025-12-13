#r "nuget: Microsoft.Data.Sqlite, 10.0.1"
#r "nuget: Dapper, 2.1.66"
#r "../bin/Debug/net9.0/Spiral.Trading.dll"

let dbPath = "../../../Python version/data/trading.db"
let conn = Spiral.Trading.Database.openConnection dbPath
Spiral.Trading.Database.initializeSchema conn
Spiral.Trading.Database.refreshMaterializedTables conn
conn.Close()
