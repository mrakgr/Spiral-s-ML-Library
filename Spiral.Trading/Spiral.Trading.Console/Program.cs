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
                Console.WriteLine("  ingest-data     Ingest downloaded CSV files into SQLite database");
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
                    case "ingest-data":
                        await RunIngestData(dbPath);
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

            await downloader.DownloadDailyAggregatesAsync(startDate, endDate, outputDir, maxDegreeOfParallelism: 8);
        }

        static async Task RunIngestData(string dbPath)
        {
            var ingestor = new DataIngestor(dbPath);
            string dataDir = "data/daily_aggregates";
            if (!Directory.Exists(dataDir))
            {
                Console.WriteLine($"Data directory not found: {dataDir}");
                return;
            }
            await ingestor.IngestDailyAggregatesAsync(dataDir);
        }

        static async Task RunSplitDownload(string apiKey, string dbPath)
        {
            Console.WriteLine("Loading tickers from database...");
            
            using var context = new TradingDbContext(dbPath);
            await context.Database.EnsureCreatedAsync();

            // query distinct tickers from DailyPrices
            var validTickers = await context.DailyPrices
                                            .Select(p => p.Ticker)
                                            .Distinct()
                                            .ToListAsync();

            if (!validTickers.Any())
            {
                Console.WriteLine("No tickers found in database. Please run 'ingest-data' first.");
                return;
            }

            Console.WriteLine($"Loaded {validTickers.Count} unique tickers from database");

            var downloader = new PolygonSplitDownloader(apiKey);
            var splits = await downloader.DownloadSplitsAsync(validTickers, maxDegreeOfParallelism: 20);

            Console.WriteLine($"\nFound {splits.Count} splits. Saving to database...");

            // Batch insert using EF Core
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
