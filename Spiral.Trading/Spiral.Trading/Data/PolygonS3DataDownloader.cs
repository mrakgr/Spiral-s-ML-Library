using Amazon;
using Amazon.S3;
using Amazon.S3.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace Spiral.Trading.Data
{
    public class PolygonS3DataDownloader : IDataDownloader
    {
        private readonly string _accessKey;
        private readonly string _secretKey;
        private readonly string _bucketName = "flatfiles";
        private readonly AmazonS3Client _s3Client;

        public PolygonS3DataDownloader(string accessKey, string secretKey)
        {
            _accessKey = accessKey;
            _secretKey = secretKey;
            
            var config = new AmazonS3Config
            {
                ServiceURL = "https://files.massive.com",
                AuthenticationRegion = "us-east-1", // Default, usually ignored with custom ServiceURL but required by SDK
                ForcePathStyle = true
            };

            _s3Client = new AmazonS3Client(_accessKey, _secretKey, config);
        }

        public async Task DownloadDailyAggregatesAsync(DateTime startDate, DateTime endDate, string outputDirectory, int maxDegreeOfParallelism = 20)
        {
            Directory.CreateDirectory(outputDirectory);
            Console.WriteLine($"Downloading daily aggregate data from {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");

            var dates = new List<DateTime>();
            var currentDate = startDate;
            while (currentDate <= endDate)
            {
                if (currentDate.DayOfWeek != DayOfWeek.Saturday && currentDate.DayOfWeek != DayOfWeek.Sunday)
                {
                    dates.Add(currentDate);
                }
                currentDate = currentDate.AddDays(1);
            }

            // Using Parallel.ForEachAsync to download in parallel
            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };
            
            int total = dates.Count;
            int completed = 0;

            await Parallel.ForEachAsync(dates, parallelOptions, async (date, ct) =>
            {
                string dateStr = date.ToString("yyyy-MM-dd");
                string year = date.Year.ToString();
                string month = date.Month.ToString("00");

                // S3 Key format: us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
                string s3Key = $"us_stocks_sip/day_aggs_v1/{year}/{month}/{dateStr}.csv.gz";
                string localFilePath = Path.Combine(outputDirectory, $"{dateStr}.csv.gz");

                bool skipped = false;
                bool success = false;
                string error = null;

                if (File.Exists(localFilePath))
                {
                    skipped = true;
                }
                else
                {
                    int maxRetries = 5;
                    for (int attempt = 1; attempt <= maxRetries; attempt++)
                    {
                        try
                        {
                            var request = new GetObjectRequest
                            {
                                BucketName = _bucketName,
                                Key = s3Key
                            };

                            using (var response = await _s3Client.GetObjectAsync(request, ct))
                            using (var responseStream = response.ResponseStream)
                            using (var fileStream = File.Create(localFilePath))
                            {
                                await responseStream.CopyToAsync(fileStream, ct);
                            }
                            success = true;
                            break; // Success, exit retry loop
                        }
                        catch (AmazonS3Exception ex) when (ex.StatusCode == System.Net.HttpStatusCode.ServiceUnavailable || 
                                                           ex.StatusCode == System.Net.HttpStatusCode.TooManyRequests || 
                                                           ex.ErrorCode == "TooManyRequests" || 
                                                           ex.ErrorCode == "SlowDown")
                        {
                            if (attempt == maxRetries)
                            {
                                error = $"S3 Error (MAX RETRIES): {ex.StatusCode} - {ex.Message}";
                            }
                            else
                            {
                                // Exponential backoff: 2s, 4s, 8s, 16s...
                                int delay = (int)Math.Pow(2, attempt) * 1000;
                                await Task.Delay(delay, ct);
                            }
                        }
                        catch (AmazonS3Exception ex)
                        {
                             // Non-retriable S3 error (e.g. 404, 403)
                             error = $"S3 Error: {ex.StatusCode} - {ex.Message}";
                             break;
                        }
                        catch (Exception ex)
                        {
                             // Other errors
                             error = ex.Message;
                             break;
                        }
                    }
                }

                // Thread-safe progress reporting
                int c = System.Threading.Interlocked.Increment(ref completed);
                
                string status = skipped ? "Skipped" : (success ? "Downloaded" : "Failed");
                string msg = $"[{c}/{total}] {dateStr}: {status}";
                if (!string.IsNullOrEmpty(error)) msg += $" ({error})";

                Console.WriteLine(msg);
            });

            Console.WriteLine("\nDownload Complete.");
        }
    }
}
