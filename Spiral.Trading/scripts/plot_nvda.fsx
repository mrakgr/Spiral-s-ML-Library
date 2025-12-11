
#r "nuget: Microsoft.Data.Sqlite, 9.0.0"
#r "nuget: Microsoft.EntityFrameworkCore, 9.0.0"
#r "nuget: Microsoft.EntityFrameworkCore.Sqlite, 9.0.0"
#r "nuget: XPlot.Plotly, 4.1.0"
#r "../Spiral.Trading/bin/Debug/net9.0/Spiral.Trading.dll"

open System
open System.IO
open System.Linq
open Microsoft.EntityFrameworkCore
open Spiral.Trading.Storage
open Spiral.Trading.Plotting

let dbPath = Path.GetFullPath("data/trading.db")
printfn "Database Path: %s" dbPath

let context = new TradingDbContext(dbPath)

let ticker = "NVDA"
printfn "Fetching data for %s..." ticker

let prices = 
    context.DailyPrices
        .Where(fun p -> p.Ticker = ticker)
        .OrderBy(fun p -> p.Date)
        .ToList()

printfn "Found %d records for %s." prices.Count ticker

if prices.Any() then
    let outputPath = sprintf "%s_chart.html" ticker
    ChartGenerator.GenerateCandlestickChart(prices, ticker, outputPath, 1600, 900)
    printfn "Chart generated: %s" (Path.GetFullPath(outputPath))
else
    printfn "No data found. Please run 'make ingest' first."
