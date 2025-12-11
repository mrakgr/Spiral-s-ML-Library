#r "nuget: XPlot.Plotly, 4.1.0"
#r "nuget: Microsoft.EntityFrameworkCore, 9.0.11"
#r "nuget: Microsoft.EntityFrameworkCore.Sqlite, 9.0.11"
#r "../Spiral.Trading/bin/Debug/net9.0/Spiral.Trading.dll"

open System.IO
open Spiral.Trading.Plotting

let dbPath = Path.GetFullPath("data/trading.db")
let ticker = "NVDA"
let outputPath = sprintf "%s_chart.html" ticker

printfn "Generating chart for %s..." ticker
StockChartService.GenerateChart(dbPath, ticker, outputPath, 1600, 900)
printfn "Chart saved to %s" (Path.GetFullPath(outputPath))
