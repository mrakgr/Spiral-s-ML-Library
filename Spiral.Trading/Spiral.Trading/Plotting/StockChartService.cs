using System;
using System.IO;
using System.Linq;
using Spiral.Trading.Storage;
using Spiral.Trading.Models;

namespace Spiral.Trading.Plotting
{
    public class StockChartService
    {
        public static void GenerateChart(string dbPath, string ticker, string outputPath, int width = 1200, int height = 900)
        {
            if (!File.Exists(dbPath))
            {
                throw new FileNotFoundException($"Database not found at {dbPath}");
            }

            using (var context = new TradingDbContext(dbPath))
            {
                var prices = context.DailyPrices
                                    .Where(p => p.Ticker == ticker)
                                    .OrderBy(p => p.Date)
                                    .ToList();

                if (!prices.Any())
                {
                    Console.WriteLine($"No data found for ticker {ticker}.");
                    return;
                }

                Console.WriteLine($"Found {prices.Count} records for {ticker}.");
                ChartGenerator.GenerateCandlestickChart(prices, ticker, outputPath, width, height);
            }
        }
    }
}
