#r "../Spiral.Trading/bin/Debug/net9.0/Spiral.Trading.dll"

// Spiral.Trading


// #r "nuget: XPlot.Plotly, 4.1.0"
// #r "nuget: Microsoft.EntityFrameworkCore.Sqlite, 9.0.11"
// #r "nuget: Microsoft.EntityFrameworkCore, 9.0.11"
// #r "../Spiral.Trading/bin/Debug/net9.0/Spiral.Trading.dll"

// using System;
// using System.Linq;
// using System.IO;
// using Microsoft.EntityFrameworkCore;
// using Spiral.Trading.Storage;
// using Spiral.Trading.Plotting;

// string dbPath = Path.GetFullPath("data/trading.db");
// Console.WriteLine($"Database Path: {dbPath}");

// using (var context = new TradingDbContext(dbPath))
// {
//     string ticker = "NVDA";
//     Console.WriteLine($"Fetching data for {ticker}...");

//     var prices = context.DailyPrices
//                         .Where(p => p.Ticker == ticker)
//                         .OrderBy(p => p.Date)
//                         .ToList();

//     Console.WriteLine($"Found {prices.Count} records for {ticker}.");

//     if (prices.Any())
//     {
//         string outputPath = $"{ticker}_chart.html";
//         ChartGenerator.GenerateCandlestickChart(prices, ticker, outputPath);
//         Console.WriteLine($"Chart generated: {Path.GetFullPath(outputPath)}");
//     }
//     else
//     {
//         Console.WriteLine("No data found. Please run 'make ingest' first.");
//     }
// }
