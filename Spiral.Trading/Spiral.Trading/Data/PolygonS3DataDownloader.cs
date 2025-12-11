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

        public async Task DownloadDailyAggregatesAsync(DateTime startDate, DateTime endDate, string outputDirectory)
        {
            Directory.CreateDirectory(outputDirectory);
            Console.WriteLine($"Downloading daily aggregate data from {startDate:yyyy-MM-dd} to {endDate:yyyy-MM-dd}");

            var currentDate = startDate;
            while (currentDate <= endDate)
            {
                // Skip weekends
                if (currentDate.DayOfWeek == DayOfWeek.Saturday || currentDate.DayOfWeek == DayOfWeek.Sunday)
                {
                    currentDate = currentDate.AddDays(1);
                    continue;
                }

                string dateStr = currentDate.ToString("yyyy-MM-dd");
                string year = currentDate.Year.ToString();
                string month = currentDate.Month.ToString("00");
                
                // S3 Key format: us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
                string s3Key = $"us_stocks_sip/day_aggs_v1/{year}/{month}/{dateStr}.csv.gz";
                string localFilePath = Path.Combine(outputDirectory, $"{dateStr}.csv.gz");

                if (File.Exists(localFilePath))
                {
                    Console.WriteLine($"\u2299 {dateStr}: Already downloaded, skipping");
                }
                else
                {
                    try
                    {
                        Console.Write($"Downloading {dateStr}... ");
                        var request = new GetObjectRequest
                        {
                            BucketName = _bucketName,
                            Key = s3Key
                        };

                        using (var response = await _s3Client.GetObjectAsync(request))
                        using (var responseStream = response.ResponseStream)
                        using (var fileStream = File.Create(localFilePath))
                        {
                            await responseStream.CopyToAsync(fileStream);
                        }
                        Console.WriteLine("\u2713");
                    }
                    catch (AmazonS3Exception ex)
                    {
                        Console.WriteLine($"\u2717 (S3 Error: {ex.Message})");
                        // 404 means file not found, likely holiday or not ready
                    }
                    catch (Exception ex)
                    {
                         Console.WriteLine($"\u2717 (Error: {ex.Message})");
                    }
                }

                currentDate = currentDate.AddDays(1);
            }
            
            Console.WriteLine("\nDownload Complete.");
        }
    }
}
