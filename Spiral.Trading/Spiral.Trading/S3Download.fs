module Spiral.Trading.S3Download

open System
open System.IO
open System.Net
open System.Threading
open Amazon.S3
open Amazon.S3.Model

/// S3 configuration for Massive
let private bucketName = "flatfiles"
let private serviceUrl = "https://files.massive.com"
let private maxRetries = 5

/// Create an S3 client configured for Massive
let createS3Client (accessKey: string) (secretKey: string) : AmazonS3Client =
    let config = AmazonS3Config(
        ServiceURL = serviceUrl,
        AuthenticationRegion = "us-east-1",
        ForcePathStyle = true
    )
    new AmazonS3Client(accessKey, secretKey, config)

/// Generate the S3 key for a daily aggregate file
let private getS3Key (date: DateTime) : string =
    let dateStr = date.ToString("yyyy-MM-dd")
    let year = date.Year.ToString()
    let month = date.Month.ToString("00")
    $"us_stocks_sip/day_aggs_v1/{year}/{month}/{dateStr}.csv.gz"

/// Generate trading days (excluding weekends) between two dates
let getTradingDays (startDate: DateTime) (endDate: DateTime) : DateTime list =
    let rec loop (current: DateTime) acc =
        if current > endDate then
            List.rev acc
        else
            let next = current.AddDays(1.0)
            if current.DayOfWeek = DayOfWeek.Saturday || current.DayOfWeek = DayOfWeek.Sunday then
                loop next acc
            else
                loop next (current :: acc)
    loop startDate []

/// Download a single day's data from S3
let private downloadSingleDay
    (client: AmazonS3Client)
    (outputDir: string)
    (date: DateTime)
    (ct: CancellationToken)
    : Async<DownloadResult> =
    async {
        let dateStr = date.ToString("yyyy-MM-dd")
        let s3Key = getS3Key date
        let localPath = Path.Combine(outputDir, $"{dateStr}.csv.gz")

        if File.Exists localPath then
            return Skipped date
        else
            let rec retry attempt =
                async {
                    try
                        let request = GetObjectRequest(
                            BucketName = bucketName,
                            Key = s3Key
                        )

                        let! response = client.GetObjectAsync(request, ct) |> Async.AwaitTask

                        use responseStream = response.ResponseStream
                        use fileStream = File.Create(localPath)
                        do! responseStream.CopyToAsync(fileStream, ct) |> Async.AwaitTask

                        return Downloaded date
                    with
                    | :? AmazonS3Exception as ex
                        when (ex.StatusCode = HttpStatusCode.ServiceUnavailable
                              || ex.StatusCode = HttpStatusCode.TooManyRequests
                              || ex.ErrorCode.Contains "TooManyRequests"
                              || ex.ErrorCode.Contains "SlowDown")
                              && attempt < maxRetries ->
                        // Exponential backoff: 2s, 4s, 8s, 16s...
                        let delay = pown 2 attempt * 1000
                        do! Async.Sleep delay
                        return! retry (attempt + 1)

                    | :? AmazonS3Exception as ex when attempt >= maxRetries ->
                        return Failed(date, $"S3 Error (MAX RETRIES): {ex.StatusCode} - {ex.Message}")

                    | :? AmazonS3Exception as ex ->
                        return Failed(date, $"S3 Error: {ex.StatusCode} - {ex.Message}")

                    | ex ->
                        return Failed(date, ex.Message)
                }
            return! retry 1
    }

/// Progress callback type
type ProgressCallback = int -> int -> DownloadResult -> unit

/// Download daily aggregates for a date range
let downloadDailyAggregates
    (client: AmazonS3Client)
    (startDate: DateTime)
    (endDate: DateTime)
    (outputDir: string)
    (maxParallelism: int)
    (progress: ProgressCallback option)
    (ct: CancellationToken)
    : Async<DownloadResult list> =
    async {
        Directory.CreateDirectory(outputDir) |> ignore

        let dates = getTradingDays startDate endDate
        let total = dates.Length
        let completed = ref 0

        let reportProgress result =
            let c = Interlocked.Increment(completed)
            match progress with
            | Some callback -> callback c total result
            | None -> ()

        // Use SemaphoreSlim for parallelism control
        use semaphore = new SemaphoreSlim(maxParallelism, maxParallelism)

        let downloadWithSemaphore date =
            async {
                do! semaphore.WaitAsync(ct) |> Async.AwaitTask
                try
                    let! result = downloadSingleDay client outputDir date ct
                    reportProgress result
                    return result
                finally
                    semaphore.Release() |> ignore
            }

        let! results =
            dates
            |> List.map downloadWithSemaphore
            |> Async.Parallel

        return results |> Array.toList
    }

/// Default progress reporter that prints to console
let consoleProgress (completed: int) (total: int) (result: DownloadResult) : unit =
    let status, dateStr =
        match result with
        | Downloaded date -> "Downloaded", date.ToString("yyyy-MM-dd")
        | Skipped date -> "Skipped", date.ToString("yyyy-MM-dd")
        | Failed (date, error) -> $"Failed ({error})", date.ToString("yyyy-MM-dd")

    printfn "[%d/%d] %s: %s" completed total dateStr status
