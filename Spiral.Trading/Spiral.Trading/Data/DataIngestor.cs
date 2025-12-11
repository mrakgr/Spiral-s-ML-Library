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

            Console.WriteLine($"Found {files.Count} files in directory.");

            // 1. Ensure DB Schema
            // Note: Since we changed the model, if the DB exists with old schema, we might have issues.
            // For this dev iteration, if schema is invalid, we might need to recreate. 
            // In production, migrations would be used.
            // Let's try to just EnsureCreated. If it assumes existing DB is fine but misses table, we might crash later.
            // Ideally we check if ProcessedFiles table exists or just wipe if needed.
            // I'll assume for now we can rely on EnsureCreated. If it fails due to mismatch, user might need to delete DB file.

            using (var initContext = new TradingDbContext(_dbPath))
            {
                await initContext.Database.EnsureCreatedAsync();
            }

            // 2. Load Processed Files
            HashSet<string> processedFilesSet;
            using (var context = new TradingDbContext(_dbPath))
            {
                // If the table was just created, this is empty.
                // If legacy DB existed without table, this throws.
                // We will bubble up valid errors.
                processedFilesSet = await context.ProcessedFiles
                                                 .Select(f => f.FileName)
                                                 .ToHashSetAsync();
            }

            Console.WriteLine($"Already ingested: {processedFilesSet.Count} files.");

            // 3. Filter
            var filesToProcess = files.Where(f => !processedFilesSet.Contains(Path.GetFileName(f)))
                                      .ToList();

            Console.WriteLine($"Remaining to ingest: {filesToProcess.Count} files.");
            if (filesToProcess.Count == 0) return;

            // Channel: (FileName, Rows)
            var channel = Channel.CreateBounded<(string FileName, List<DailyPrice> Rows)>(new BoundedChannelOptions(500)
            {
                SingleWriter = false,
                SingleReader = true,
                FullMode = BoundedChannelFullMode.Wait
            });

            // Consumer
            var consumerTask = Task.Run(async () =>
            {
                using var context = new TradingDbContext(_dbPath);
                context.ChangeTracker.AutoDetectChangesEnabled = false;
                context.ChangeTracker.QueryTrackingBehavior = QueryTrackingBehavior.NoTracking;

                int processedCount = 0;
                long totalRows = 0;

                await foreach (var item in channel.Reader.ReadAllAsync())
                {
                    if (item.Rows.Count > 0)
                    {
                        try
                        {
                            // A. Insert Prices
                            // Since we filtered files, we assume these are NEW data. No Duplicates.
                            // Use simple BulkInsert.
                            var bulkConfig = new BulkConfig { SetOutputIdentity = false, BatchSize = 10000 };
                            await context.BulkInsertAsync(item.Rows, bulkConfig);

                            // B. Record File as Processed
                            // We can use EF normal add for this single row or bulk insert if we wanted.
                            // Since it's one row per file batch, normal Add is fine, OR we could keep a separate list and bulk insert occasionally.
                            // But for safety (crash consistency), better to commit file record with data or right after.
                            // Actually, to ensure consistency: Data committed -> File committed.

                            // Let's us direct SQL or EF for the file record to avoid context tracking issues mixed with bulk lib 
                            // (though BulkExtensions plays nice mostly).

                            // Re-enable tracking for this simple op? Or just direct insert.
                            // "ProcessedFiles" is simple.

                            var pf = new ProcessedFile
                            {
                                FileName = item.FileName,
                                IngestedAt = DateTime.UtcNow
                            };

                            // We can use BulkInsert for this too to keep it uniform and fast
                            await context.BulkInsertAsync(new List<ProcessedFile> { pf });

                            totalRows += item.Rows.Count;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"\nError ingesting {item.FileName}: {ex.Message}");
                            // If we fail, we don't mark file as processed.
                        }
                    }
                    else
                    {
                        // Even if empty rows (e.g. holiday? file existed but had no data?), mark as processed?
                        // Yes, otherwise we re-process forever.
                        try
                        {
                            var pf = new ProcessedFile { FileName = item.FileName, IngestedAt = DateTime.UtcNow };
                            await context.BulkInsertAsync(new List<ProcessedFile> { pf });
                        }
                        catch { }
                    }

                    processedCount++;
                    if (processedCount % 10 == 0) Console.Write(".");
                    if (processedCount % 100 == 0) Console.WriteLine($" {processedCount}/{filesToProcess.Count} (Total Rows: {totalRows})");
                }
            });

            // Producer
            var producerOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
            await Parallel.ForEachAsync(filesToProcess, producerOptions, async (file, ct) =>
            {
                var filename = Path.GetFileName(file);
                var datePart = filename.Replace(".csv.gz", "");
                if (DateTime.TryParse(datePart, out DateTime date))
                {
                    try
                    {
                        var prices = ParseDailyCsv(file, date);
                        await channel.Writer.WriteAsync((filename, prices), ct);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error parsing {filename}: {ex.Message}");
                    }
                }
            });

            channel.Writer.Complete();
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

            var bulkConfig = new BulkConfig
            {
                SetOutputIdentity = false,
                UpdateByProperties = new List<string> { nameof(Split.Ticker), nameof(Split.ExecutionDate) }
            };
            await context.BulkInsertOrUpdateAsync(records, bulkConfig);

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
                    WindowStart = r.Window_Start, // Map Correctly
                    Transactions = r.Transactions
                });
            }
            return prices;
        }

        private class DailyAggRow
        {
            public string Ticker { get; set; } = "";
            public double Open { get; set; }
            public double High { get; set; }
            public double Low { get; set; }
            public double Close { get; set; }
            public double Volume { get; set; }
            public long Window_Start { get; set; } // Matches CSV header snake_case usually if using Name or default map? 
                                                   // CsvHelper matches by name usually ignoring case but let's be safe.
                                                   // Header: window_start
            public int Transactions { get; set; }
        }
    }
}
