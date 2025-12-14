module Spiral.Trading.IntradayDownload

open System
open System.IO
open System.Net
open System.Net.Http
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading

/// Timespan for aggregate bars
type AggregateTimespan =
    | Second
    | Minute

    member this.ToApiString() =
        match this with
        | Second -> "second"
        | Minute -> "minute"

/// A single aggregate bar from the Polygon API
[<CLIMutable>]
type AggregateBar = {
    [<JsonPropertyName("o")>]
    Open: float

    [<JsonPropertyName("h")>]
    High: float

    [<JsonPropertyName("l")>]
    Low: float

    [<JsonPropertyName("c")>]
    Close: float

    [<JsonPropertyName("v")>]
    Volume: float

    [<JsonPropertyName("vw")>]
    Vwap: float option

    [<JsonPropertyName("t")>]
    Timestamp: int64

    [<JsonPropertyName("n")>]
    Transactions: int option
}

/// Response from the Polygon aggregates API
[<CLIMutable>]
type AggregatesResponse = {
    [<JsonPropertyName("ticker")>]
    Ticker: string

    [<JsonPropertyName("queryCount")>]
    QueryCount: int

    [<JsonPropertyName("resultsCount")>]
    ResultsCount: int

    [<JsonPropertyName("adjusted")>]
    Adjusted: bool

    [<JsonPropertyName("results")>]
    Results: AggregateBar[] option

    [<JsonPropertyName("status")>]
    Status: string

    [<JsonPropertyName("request_id")>]
    RequestId: string

    [<JsonPropertyName("next_url")>]
    NextUrl: string option
}

/// Result of downloading intraday data for a ticker/date
type IntradayDownloadResult =
    | IntradayDownloaded of ticker: string * date: DateTime * barCount: int
    | IntradaySkipped of ticker: string * date: DateTime
    | IntradayFailed of ticker: string * date: DateTime * error: string

let private baseUrl = "https://api.polygon.io/v2/aggs/ticker"
let private maxRetries = 5
let private baseDelayMs = 200

let private jsonOptions =
    let options = JsonSerializerOptions()
    options.PropertyNameCaseInsensitive <- true
    options

/// Generate output file path for intraday data
let private getOutputPath (outputDir: string) (ticker: string) (date: DateTime) (timespan: AggregateTimespan) : string =
    let timespanDir = timespan.ToApiString()
    let dateStr = date.ToString("yyyy-MM-dd")
    let dir = Path.Combine(outputDir, timespanDir, ticker)
    Directory.CreateDirectory(dir) |> ignore
    Path.Combine(dir, $"{dateStr}.json")

/// Download aggregate bars for a single ticker and date
let downloadAggregates
    (httpClient: HttpClient)
    (apiKey: string)
    (ticker: string)
    (date: DateTime)
    (timespan: AggregateTimespan)
    (ct: CancellationToken)
    : Async<Result<AggregatesResponse, string>> =
    async {
        let dateStr = date.ToString("yyyy-MM-dd")
        let timespanStr = timespan.ToApiString()

        // Build URL: /v2/aggs/ticker/{ticker}/range/1/{timespan}/{from}/{to}
        let url = $"{baseUrl}/{ticker}/range/1/{timespanStr}/{dateStr}/{dateStr}?adjusted=true&sort=asc&limit=50000&apiKey={apiKey}"

        let rec retry attempt delay =
            async {
                try
                    let! response = httpClient.GetAsync(url, ct) |> Async.AwaitTask

                    if response.StatusCode = HttpStatusCode.TooManyRequests then
                        if attempt >= maxRetries then
                            return Error "Rate limited: max retries exceeded"
                        else
                            let jitter = Random.Shared.Next(10, 100)
                            let currentDelay = delay + jitter
                            do! Async.Sleep currentDelay
                            let newDelay = int (float delay * 1.5)
                            return! retry (attempt + 1) newDelay
                    elif response.StatusCode = HttpStatusCode.NotFound then
                        return Error $"No data found for {ticker} on {dateStr}"
                    else
                        response.EnsureSuccessStatusCode() |> ignore
                        let! content = response.Content.ReadAsStringAsync(ct) |> Async.AwaitTask
                        let parsed = JsonSerializer.Deserialize<AggregatesResponse>(content, jsonOptions)
                        return Ok parsed
                with
                | :? HttpRequestException as ex when attempt < maxRetries ->
                    do! Async.Sleep 500
                    return! retry (attempt + 1) delay
                | ex ->
                    return Error ex.Message
            }

        return! retry 0 baseDelayMs
    }

/// Download and save aggregate bars for a single ticker and date
let downloadAndSaveAggregates
    (httpClient: HttpClient)
    (apiKey: string)
    (outputDir: string)
    (ticker: string)
    (date: DateTime)
    (timespan: AggregateTimespan)
    (ct: CancellationToken)
    : Async<IntradayDownloadResult> =
    async {
        let outputPath = getOutputPath outputDir ticker date timespan

        if File.Exists outputPath then
            return IntradaySkipped(ticker, date)
        else
            let! result = downloadAggregates httpClient apiKey ticker date timespan ct

            match result with
            | Error msg ->
                return IntradayFailed(ticker, date, msg)
            | Ok response ->
                let barCount = response.Results |> Option.map Array.length |> Option.defaultValue 0

                // Save the raw response as JSON
                let json = JsonSerializer.Serialize(response, jsonOptions)
                do! File.WriteAllTextAsync(outputPath, json, ct) |> Async.AwaitTask

                return IntradayDownloaded(ticker, date, barCount)
    }

/// Progress callback type for intraday downloads
type IntradayProgressCallback = int -> int -> IntradayDownloadResult -> unit

/// Download intraday data for multiple ticker/date pairs
let downloadIntradayBatch
    (httpClient: HttpClient)
    (apiKey: string)
    (outputDir: string)
    (tickerDates: (string * DateTime) list)
    (timespan: AggregateTimespan)
    (maxParallelism: int)
    (progress: IntradayProgressCallback option)
    (ct: CancellationToken)
    : Async<IntradayDownloadResult list> =
    async {
        let total = tickerDates.Length
        let completed = ref 0

        let reportProgress result =
            let c = Threading.Interlocked.Increment(completed)
            match progress with
            | Some callback -> callback c total result
            | None -> ()

        // Use SemaphoreSlim for parallelism control
        use semaphore = new SemaphoreSlim(maxParallelism, maxParallelism)

        let downloadWithSemaphore (ticker, date) =
            async {
                do! semaphore.WaitAsync(ct) |> Async.AwaitTask
                try
                    let! result = downloadAndSaveAggregates httpClient apiKey outputDir ticker date timespan ct
                    reportProgress result
                    return result
                finally
                    semaphore.Release() |> ignore
            }

        let! results =
            tickerDates
            |> List.map downloadWithSemaphore
            |> Async.Parallel

        return results |> Array.toList
    }

/// Default progress reporter that prints to console
let consoleProgress (completed: int) (total: int) (result: IntradayDownloadResult) : unit =
    let formatDate (d: DateTime) = d.ToString("yyyy-MM-dd")
    let status =
        match result with
        | IntradayDownloaded(ticker, date, bars) ->
            $"Downloaded {ticker} {formatDate date} ({bars} bars)"
        | IntradaySkipped(ticker, date) ->
            $"Skipped {ticker} {formatDate date} (exists)"
        | IntradayFailed(ticker, date, error) ->
            $"Failed {ticker} {formatDate date}: {error}"

    printfn "[%d/%d] %s" completed total status
