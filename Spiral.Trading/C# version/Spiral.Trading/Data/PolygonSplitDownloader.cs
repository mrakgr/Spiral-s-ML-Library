using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
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
                    // next_url from Polygon already includes the apiKey parameter
                    url = nextUrl.Contains("apiKey=") ? nextUrl : $"{nextUrl}&apiKey={_apiKey}";
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
