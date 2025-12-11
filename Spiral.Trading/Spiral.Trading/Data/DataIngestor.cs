using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.EntityFrameworkCore;
using Spiral.Trading.Models;
using Spiral.Trading.Storage;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;

namespace Spiral.Trading.Data
{
    public class DataIngestor
    {
        private readonly string _dbPath;

        public DataIngestor(string dbPath)
        {
            _dbPath = dbPath;
        }

        public async Task IngestDailyAggregatesAsync(string dataDirectory)
        {
            var files = Directory.GetFiles(dataDirectory, "*.csv.gz")
                                 .OrderBy(f => f)
                                 .ToList();

            Console.WriteLine($"Found {files.Count} files to ingest.");

            using var context = new TradingDbContext(_dbPath);
            await context.Database.EnsureCreatedAsync();

            int processed = 0;
            int batchSize = 10;

            for (int i = 0; i < files.Count; i += batchSize)
            {
                var batchFiles = files.Skip(i).Take(batchSize).ToList();
                using var transaction = await context.Database.BeginTransactionAsync();

                try
                {
                    foreach (var file in batchFiles)
                    {
                        var filename = Path.GetFileName(file);
                        var datePart = filename.Replace(".csv.gz", "");
                        if (!DateTime.TryParse(datePart, out DateTime date))
                        {
                            Console.WriteLine($"Skipping {filename}: Cannot parse date.");
                            continue;
                        }

                        // Check if data for this date already exists to avoid duplication errors
                        // Optimization: Check simply if any record exists for this date, assume complete if so?
                        // Or just try/catch unique violations per file? Checking count is safer.
                        bool exists = await context.DailyPrices.AnyAsync(p => p.Date == date);
                        if (exists)
                        {
                            // Console.WriteLine($"Skipping {filename}: Data for {date:yyyy-MM-dd} already exists.");
                            continue;
                        }

                        try
                        {
                            var prices = ParseDailyCsv(file, date);
                            await context.DailyPrices.AddRangeAsync(prices);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error parsing {filename}: {ex.Message}");
                        }

                        processed++;
                        if (processed % 10 == 0) Console.Write(".");
                        if (processed % 100 == 0) Console.WriteLine($" {processed}/{files.Count}");
                    }

                    await context.SaveChangesAsync();
                    await transaction.CommitAsync();

                    // Detach entities to free memory
                    context.ChangeTracker.Clear();
                }
                catch (Exception ex)
                {
                    await transaction.RollbackAsync();
                    Console.WriteLine($"\nError processing batch starting at index {i}: {ex.Message}");
                    // Re-throw or continue? If we continue we might skip a chunk.
                    // Let's log and continue to next batch.
                }
            }
            Console.WriteLine("\nIngestion Complete.");
        }

        public async Task IngestSplitsAsync(string splitsFilePath)
        {
            if (!File.Exists(splitsFilePath))
            {
                Console.WriteLine($"Splits file not found: {splitsFilePath}");
                return;
            }

            Console.WriteLine("Ingesting splits...");
            using var reader = new StreamReader(splitsFilePath);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);

            var records = csv.GetRecords<Split>().ToList();

            using var context = new TradingDbContext(_dbPath);
            await context.Database.EnsureCreatedAsync();

            await context.Splits.AddRangeAsync(records);
            await context.SaveChangesAsync();

            Console.WriteLine($"Ingested {records.Count} splits.");
        }

        private List<DailyPrice> ParseDailyCsv(string filePath, DateTime date)
        {
            using var fileStream = File.OpenRead(filePath);
            using var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress);
            using var reader = new StreamReader(gzipStream);
            using var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
                MissingFieldFound = null,
                HeaderValidated = null
            });

            var records = csv.GetRecords<DailyAggRow>().ToList();

            return records.Select(r => new DailyPrice
            {
                Ticker = r.Ticker,
                Date = date,
                Open = r.Open,
                High = r.High,
                Low = r.Low,
                Close = r.Close,
                Volume = (long)r.Volume,
                Transactions = r.Transactions
            }).ToList();
        }

        private class DailyAggRow
        {
            public string Ticker { get; set; }
            public double Open { get; set; }
            public double High { get; set; }
            public double Low { get; set; }
            public double Close { get; set; }
            public double Volume { get; set; }
            public int Transactions { get; set; }
        }
    }
}
