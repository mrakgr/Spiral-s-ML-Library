using Microsoft.EntityFrameworkCore;
using Spiral.Trading.Config;
using Spiral.Trading.Data;
using Spiral.Trading.Storage;
using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace Spiral.Trading.ConsoleApp
{
    class Program
    {
        static async Task Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: dotnet run -- [command]");
                Console.WriteLine("Commands:");
                Console.WriteLine("  download-bulk   Download daily aggregate files from S3 to disk");
                Console.WriteLine("  download-splits Download stock splits from Polygon API to Database");
                return;
            }

            string command = args[0].ToLower();

            // Load Configuration
            string apiKeyPath = Path.Combine(Environment.CurrentDirectory, "api_key.json");
            if (!File.Exists(apiKeyPath))
            {
                throw new FileNotFoundException("API key file not found. Please ensure 'api_key.json' is in the current directory or the project root.", apiKeyPath);
            }

            try 
            {
                var (apiKey, s3Access, s3Secret) = ConfigLoader.LoadKeys(apiKeyPath);
                
                // Initialize Database
                string dbPath = "data/trading.db";
                Directory.CreateDirectory("data");
                
                using (var context = new TradingDbContext(dbPath))
                {
                    await context.Database.EnsureCreatedAsync();
                }

                switch (command)
                {
                    case "download-bulk":
                        await RunBulkDownload(s3Access, s3Secret);
                        break;
                    case "download-splits":
                        await RunSplitDownload(apiKey, dbPath);
                        break;
                    default:
                        Console.WriteLine($"Unknown command: {command}");
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }

        static async Task RunBulkDownload(string accessKey, string secretKey)
        {
            var downloader = new PolygonS3DataDownloader(accessKey, secretKey);
            
            // Hardcoded n=5 years from now, similar to Python script
            var endDate = DateTime.Now;
            var startDate = endDate.AddYears(-5);
            
            // Assume running from project root
            string outputDir = "data/daily_aggregates";
            
            Console.WriteLine($"Output Directory: {outputDir}");

            await downloader.DownloadDailyAggregatesAsync(startDate, endDate, outputDir, maxDegreeOfParallelism: 30);
        }

        static async Task RunSplitDownload(string apiKey, string dbPath)
        {
            // Load tickers
            string tickerPath = "data/all_tickers.txt";
            if (!File.Exists(tickerPath))
            {
                // Fallback attempt or just error out if we strictly assume root
                Console.WriteLine($"Warning: {tickerPath} not found in current directory.");
            }

            if (!File.Exists(tickerPath))
            {
                Console.WriteLine($"Error: Ticker file not found at {tickerPath}");
                return;
            }

            Console.WriteLine("Loading tickers...");
            var tickers = await File.ReadAllLinesAsync(tickerPath);
            var validTickers = tickers.Where(t => !string.IsNullOrWhiteSpace(t)).ToList();
            Console.WriteLine($"Loaded {validTickers.Count} unique tickers");

            var downloader = new PolygonSplitDownloader(apiKey);
            var splits = await downloader.DownloadSplitsAsync(validTickers, maxDegreeOfParallelism: 20);

            Console.WriteLine($"\nFound {splits.Count} splits. Saving to database...");

            using var context = new TradingDbContext(dbPath);
            await context.Database.EnsureCreatedAsync(); // Ensure DB exists

            // Batch insert using EF Core
            // Just add range and save changes. For massive amounts, bulk extensions are better but for splits (usually < 20k rows) this is fine.
            // We need to handle potential duplicates if strict unique constraints are on.
            // EF Core default behavior on duplicate key is exception. 
            // We can check existence or clear table? Python script behavior was to load CSV.
            // Let's filter out existing ones locally or just try insert.
            // Best approach for idempotency: Get existing splits from DB, filter, insert new.
            
            var existingSplits = await context.Splits
                .Select(s => new { s.Ticker, s.ExecutionDate })
                .ToListAsync();
                
            var existingSet = existingSplits.Select(x => $"{x.Ticker}|{x.ExecutionDate}").ToHashSet();
            
            var newSplits = splits.Where(s => !existingSet.Contains($"{s.Ticker}|{s.ExecutionDate}")).ToList();
            
            if (newSplits.Any())
            {
                await context.Splits.AddRangeAsync(newSplits);
                await context.SaveChangesAsync();
                Console.WriteLine($"Saved {newSplits.Count} new splits.");
            }
            else
            {
                Console.WriteLine("No new splits to save.");
            }
        }
    }
}
