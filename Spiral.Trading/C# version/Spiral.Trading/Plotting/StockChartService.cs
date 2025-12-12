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

                var splits = context.Splits
                                    .Where(s => s.Ticker == ticker)
                                    .OrderByDescending(s => s.ExecutionDate)
                                    .ToList();

                if (splits.Count != 0)
                {
                    Console.WriteLine($"Found {splits.Count} splits for {ticker}. Adjusting prices...");

                    // We process from newest to oldest to accumulate the multiplier.
                    // However, 'prices' is currently sorted by Date (Oldest -> Newest).
                    // We can iterate backwards or just use a helper method.
                    // Let's iterate backwards through the list.

                    double currentMultiplier = 1.0;
                    int splitIndex = 0;

                    // Iterate from end (Newest) to beginning (Oldest)
                    for (int i = prices.Count - 1; i >= 0; i--)
                    {
                        var price = prices[i];

                        // Check if we passed any split dates (going backwards in time)
                        // Verify checks: price.Date is the date of the candle.
                        // Split.ExecutionDate: The date the split is effective.
                        // If price.Date < Split.ExecutionDate, that price is "pre-split" and needs adjustment.

                        while (splitIndex < splits.Count && price.Date < splits[splitIndex].ExecutionDate)
                        {
                            var s = splits[splitIndex];
                            // If 1 share becomes 10 (10:1 split), From=1, To=10.
                            // Old price (e.g. 1000) needs to become 100.
                            // Multiplier: 1/10.
                            currentMultiplier *= (s.SplitFrom / s.SplitTo);
                            splitIndex++;
                        }

                        if (currentMultiplier != 1.0)
                        {
                            price.Open *= currentMultiplier;
                            price.High *= currentMultiplier;
                            price.Low *= currentMultiplier;
                            price.Close *= currentMultiplier;
                            // Volume usually goes UP, so divide by multiplier (which is < 1 for forward splits)
                            price.Volume = (long)(price.Volume / currentMultiplier);
                        }
                    }
                }

                ChartGenerator.GenerateCandlestickChart(prices, ticker, outputPath, width, height);
            }
        }
    }
}
