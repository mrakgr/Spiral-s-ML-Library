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

        public async Task<List<Split>> DownloadAllSplitsAsync(DateTime startDate, DateTime? endDate = null)
        {
            var allSplits = new List<Split>();
            var startDateStr = startDate.ToString("yyyy-MM-dd");
            var endDateStr = endDate?.ToString("yyyy-MM-dd");

            Console.WriteLine($"Downloading all splits from {startDateStr}{(endDate.HasValue ? $" to {endDateStr}" : "")}...");

            string? nextUrl = null;
            int pageCount = 0;

            do
            {
                string url;
                if (nextUrl != null)
                {
                    url = nextUrl;
                }
                else
                {
                    url = $"https://api.polygon.io/v3/reference/splits?execution_date.gte={startDateStr}";
                    if (endDate.HasValue)
                    {
                        url += $"&execution_date.lte={endDateStr}";
                    }
                    url += $"&limit=1000&apiKey={_apiKey}";
                }

                int retries = 0;
                TimeSpan delay = TimeSpan.FromMilliseconds(200);
                PolygonSplitResponse? splitResponse = null;

                while (retries < 5)
                {
                    try
                    {
                        var response = await _httpClient.GetAsync(url);

                        if (response.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
                        {
                            retries++;
                            var jitter = Random.Shared.Next(10, 100);
                            var currentDelay = delay.TotalMilliseconds + jitter;
                            Console.WriteLine($"Rate limited, retrying in {currentDelay}ms...");
                            await Task.Delay((int)currentDelay);
                            delay = delay * 1.5;
                            continue;
                        }

                        response.EnsureSuccessStatusCode();
                        splitResponse = await response.Content.ReadFromJsonAsync<PolygonSplitResponse>();
                        break;
                    }
                    catch (Exception ex)
                    {
                        if (retries >= 4)
                        {
                            Console.WriteLine($"Failed after {retries} retries: {ex.Message}");
                            throw;
                        }
                        retries++;
                        await Task.Delay(500);
                    }
                }

                if (splitResponse?.Results != null)
                {
                    foreach (var result in splitResponse.Results)
                    {
                        if (result.SplitFrom != 0 && !string.IsNullOrEmpty(result.Ticker))
                        {
                            allSplits.Add(new Split
                            {
                                Ticker = result.Ticker,
                                ExecutionDate = DateTime.Parse(result.ExecutionDate),
                                SplitFrom = result.SplitFrom,
                                SplitTo = result.SplitTo,
                                SplitRatio = result.SplitTo / result.SplitFrom
                            });
                        }
                    }
                }

                nextUrl = splitResponse?.NextUrl;
                pageCount++;

                if (splitResponse?.Results != null)
                {
                    Console.WriteLine($"Page {pageCount}: Downloaded {splitResponse.Results.Count} splits (Total: {allSplits.Count})");
                }

            } while (nextUrl != null);

            Console.WriteLine($"Downloaded {allSplits.Count} total splits across {pageCount} pages.");
            return allSplits;
        }

        public async Task<List<Split>> DownloadSplitsAsync(IEnumerable<string> tickers, int maxDegreeOfParallelism = 100)
        {
            var allSplits = new ConcurrentBag<Split>();
            var failedTickers = new ConcurrentBag<string>();
            var tickerList = tickers.ToList();
            int total = tickerList.Count;
            int processed = 0;

            Console.WriteLine($"Starting split download for {total} tickers with parallelism {maxDegreeOfParallelism}...");

            var options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };

            // We use a manual retry loop or could use Polly. 
            // Simple manual retry for 429 is sufficient here.
            await Parallel.ForEachAsync(tickerList, options, async (ticker, ct) =>
            {
                int retries = 0;
                TimeSpan delay = TimeSpan.FromMilliseconds(200);

                while (retries < 5)
                {
                    try
                    {
                        string startDate = "1990-01-01";
                        string url = $"https://api.polygon.io/v3/reference/splits?ticker={ticker}&execution_date.gte={startDate}&limit=1000&apiKey={_apiKey}";

                        var response = await _httpClient.GetAsync(url, ct);

                        if (response.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
                        {
                            retries++;
                            // Exponential backoff with jitter
                            var jitter = Random.Shared.Next(10, 100);
                            var currentDelay = delay.TotalMilliseconds + jitter;
                            await Task.Delay((int)currentDelay, ct);
                            delay = delay * 1.5; // Multiply delay
                            continue; // Retry
                        }

                        response.EnsureSuccessStatusCode();

                        var splitResponse = await response.Content.ReadFromJsonAsync<PolygonSplitResponse>(cancellationToken: ct);

                        if (splitResponse?.Results != null)
                        {
                            foreach (var result in splitResponse.Results)
                            {
                                if (result.SplitFrom != 0)
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
                        break; // Success, exit loop
                    }
                    catch (Exception ex)
                    {
                        if (retries >= 4)
                        {
                            failedTickers.Add($"{ticker}: {ex.Message}");
                        }
                        else
                        {
                            // On other network errors, maybe retry? 
                            // For now only retry on expected transient or 429.
                            // If simple exception, maybe just log and move on to avoid stalling?
                            // Let's retry on HttpRequestException too briefly
                            retries++;
                            await Task.Delay(500, ct);
                            continue;
                        }
                        break;
                    }
                }

                int p = System.Threading.Interlocked.Increment(ref processed);
                if (p % 100 == 0) Console.Write(".");
                if (p % 1000 == 0) Console.WriteLine($" [{p}/{total}]");
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

            [JsonPropertyName("next_url")]
            public string? NextUrl { get; set; }
        }

        private class PolygonSplitResult
        {
            [JsonPropertyName("ticker")]
            public string Ticker { get; set; } = "";

            [JsonPropertyName("execution_date")]
            public string ExecutionDate { get; set; } = "";

            [JsonPropertyName("split_from")]
            public double SplitFrom { get; set; }

            [JsonPropertyName("split_to")]
            public double SplitTo { get; set; }
        }
    }
}
