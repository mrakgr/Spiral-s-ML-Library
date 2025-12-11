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
            foreach (var file in files)
            {
                var filename = Path.GetFileName(file);
                var datePart = filename.Replace(".csv.gz", "");
                if (!DateTime.TryParse(datePart, out DateTime date))
                {
                    Console.WriteLine($"Skipping {filename}: Cannot parse date.");
                    continue;
                }

                try
                {
                    var prices = ParseDailyCsv(file, date);
                    // Batch insert
                    await context.DailyPrices.AddRangeAsync(prices);
                    await context.SaveChangesAsync();
                    
                    processed++;
                    if (processed % 10 == 0) Console.Write(".");
                    if (processed % 100 == 0) Console.WriteLine($" {processed}/{files.Count}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing {filename}: {ex.Message}");
                    // Reset context state if needed or continue
                    context.ChangeTracker.Clear();
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
