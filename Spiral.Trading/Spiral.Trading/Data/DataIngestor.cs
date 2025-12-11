using CsvHelper;
using CsvHelper.Configuration;
using EFCore.BulkExtensions;
using Microsoft.EntityFrameworkCore;
using Spiral.Trading.Models;
using Spiral.Trading.Storage;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Channels;
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

            // Ensure DB created
            using (var initContext = new TradingDbContext(_dbPath))
            {
                await initContext.Database.EnsureCreatedAsync();
            }

            // Channel for decoupled parsing and writing
            // Bounded to avoid exhausted memory if parsing is much faster than writing
            var channel = Channel.CreateBounded<List<DailyPrice>>(new BoundedChannelOptions(500)
            {
                SingleWriter = false,
                SingleReader = true,
                FullMode = BoundedChannelFullMode.Wait
            });

            // Consumer Task (Writer)
            var consumerTask = Task.Run(async () =>
            {
                int processedFiles = 0;
                long totalRows = 0;
                using var context = new TradingDbContext(_dbPath);

                // Bulk extensions usually doesn't use ChangeTracker, but good practice to keep it clean
                context.ChangeTracker.AutoDetectChangesEnabled = false;
                context.ChangeTracker.QueryTrackingBehavior = QueryTrackingBehavior.NoTracking;

                await foreach (var batch in channel.Reader.ReadAllAsync())
                {
                    if (batch.Count > 0)
                    {
                        try
                        {
                            // BulkInsert optimized for SQLite
                            var bulkConfig = new BulkConfig
                            {
                                SetOutputIdentity = false,
                                BatchSize = 10000 // Adjust based on memory/performance
                            };

                            await context.BulkInsertAsync(batch, bulkConfig);

                            totalRows += batch.Count;

                            context.ChangeTracker.Clear();
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"\nError writing batch (Rows: {batch.Count}): {ex.Message}");
                            context.ChangeTracker.Clear();
                        }
                    }
                    processedFiles++;
                    if (processedFiles % 10 == 0) Console.Write(".");
                    if (processedFiles % 100 == 0) Console.WriteLine($" {processedFiles}/{files.Count} (Total Rows: {totalRows})");
                }
            });

            // Producer Task (Parsers)
            // Use Parallel.ForEach to parse files concurrently
            var producerOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

            await Parallel.ForEachAsync(files, producerOptions, async (file, ct) =>
            {
                var filename = Path.GetFileName(file);
                var datePart = filename.Replace(".csv.gz", "");
                if (!DateTime.TryParse(datePart, out DateTime date))
                {
                    return;
                }

                // Skip check for speed - user can manage duplicates or we can handle unique constraint violations (slower)
                try
                {
                    var prices = ParseDailyCsv(file, date);
                    // Write even empty lists to count progress
                    await channel.Writer.WriteAsync(prices, ct);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error parsing {filename}: {ex.Message}");
                    // Write empty to count progress
                    await channel.Writer.WriteAsync(new List<DailyPrice>(), ct);
                }
            });

            // Signal end of production
            channel.Writer.Complete();

            // Wait for consumer to finish
            await consumerTask;

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

            // Could use bulk insert here too if list is large
            var bulkConfig = new BulkConfig { SetOutputIdentity = false };
            await context.BulkInsertAsync(records, bulkConfig);

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

            // Map row to model
            var prices = new List<DailyPrice>(records.Count);
            foreach (var r in records)
            {
                prices.Add(new DailyPrice
                {
                    Ticker = r.Ticker,
                    Date = date,
                    Open = r.Open,
                    High = r.High,
                    Low = r.Low,
                    Close = r.Close,
                    Volume = (long)r.Volume,
                    Transactions = r.Transactions
                });
            }
            return prices;
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
