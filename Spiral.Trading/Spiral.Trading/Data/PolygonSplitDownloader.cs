using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Spiral.Trading.Models;

namespace Spiral.Trading.Data
{
    public class PolygonSplitDownloader : ISplitDownloader
    {
        private readonly HttpClient _httpClient;
        private readonly string _apiKey;

        public PolygonSplitDownloader(string apiKey, HttpClient? httpClient = null)
        {
            _apiKey = apiKey;
            _httpClient = httpClient ?? new HttpClient();
        }

        public async Task<List<Split>> DownloadSplitsAsync(IEnumerable<string> tickers, int maxDegreeOfParallelism = 20)
        {
            var allSplits = new ConcurrentBag<Split>();
            var failedTickers = new ConcurrentBag<string>();
            var tickerList = tickers.ToList();
            int total = tickerList.Count;
            int processed = 0;
            
            Console.WriteLine($"Starting split download for {total} tickers with parallelism {maxDegreeOfParallelism}...");

            var options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };

            await Parallel.ForEachAsync(tickerList, options, async (ticker, ct) =>
            {
                try
                {
                    // Date range hardcoded to match Python script logic or can be param.
                    // Python script: start_date="2020-12-09", end_date=now
                    // But usually we want all relevant history. Let's use a wide range.
                    // Python script logic: execution_date.gte=2020-12-09
                    // I will stick to a reasonable default but maybe this should be configurable.
                    // For now, I'll match the Python script's "2020-12-09" start date as a default but maybe ask user?
                    // Actually, let's use a more inclusive start date or today's date minus 10 years if not specified.
                    // The Python script specifically targeted recent data. I'll use "2000-01-01" to be safe and get more history, 
                    // or stick to the script. The script said "Date range for split data: 2020-12-09".
                    // I'll stick to 2000-01-01 to ensure we get enough history for momentum lookbacks unless user wants exactly that.
                    // Better yet, I'll define it as a constant for now.
                    
                    string startDate = "1990-01-01"; // Safe default for "all history"
                    string url = $"https://api.polygon.io/v3/reference/splits?ticker={ticker}&execution_date.gte={startDate}&limit=1000&apiKey={_apiKey}";

                    var response = await _httpClient.GetFromJsonAsync<PolygonSplitResponse>(url, ct);

                    if (response?.Results != null)
                    {
                        foreach (var result in response.Results)
                        {
                            if (result.SplitFrom != 0) // Avoid divide by zero
                            {
                                allSplits.Add(new Split
                                {
                                    Ticker = ticker,
                                    ExecutionDate = DateTime.Parse(result.ExecutionDate),
                                    SplitFrom = result.SplitFrom,
                                    SplitTo = result.SplitTo,
                                    SplitRatio = result.SplitTo / result.SplitFrom
                                });
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    failedTickers.Add($"{ticker}: {ex.Message}");
                }
                finally
                {
                    int p = System.Threading.Interlocked.Increment(ref processed);
                    if (p % 100 == 0) Console.WriteLine($"[{p}/{total}] processed...");
                }
            });

            if (!failedTickers.IsEmpty)
            {
                Console.WriteLine($"\nFailed tickers ({failedTickers.Count}):");
                foreach (var f in failedTickers.Take(10)) Console.WriteLine($"  {f}");
                if (failedTickers.Count > 10) Console.WriteLine("  ...");
            }

            return allSplits.ToList();
        }

        private class PolygonSplitResponse
        {
            [JsonPropertyName("results")]
            public List<PolygonSplitResult>? Results { get; set; }
        }

        private class PolygonSplitResult
        {
            [JsonPropertyName("execution_date")]
            public string ExecutionDate { get; set; } = "";

            [JsonPropertyName("split_from")]
            public double SplitFrom { get; set; }

            [JsonPropertyName("split_to")]
            public double SplitTo { get; set; }
        }
    }
}
