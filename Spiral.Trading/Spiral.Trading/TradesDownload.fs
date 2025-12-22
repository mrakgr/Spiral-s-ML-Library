module Spiral.Trading.TradesDownload

open System
open System.IO
open System.Net
open System.Net.Http
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading

/// A single trade from the Polygon API
[<CLIMutable>]
type Trade = {
    [<JsonPropertyName("conditions")>]
    Conditions: int[] option

    [<JsonPropertyName("exchange")>]
    Exchange: int

    [<JsonPropertyName("id")>]
    Id: string

    [<JsonPropertyName("participant_timestamp")>]
    ParticipantTimestamp: int64

    [<JsonPropertyName("price")>]
    Price: float

    [<JsonPropertyName("sequence_number")>]
    SequenceNumber: int64

    [<JsonPropertyName("sip_timestamp")>]
    SipTimestamp: int64

    [<JsonPropertyName("size")>]
    Size: float

    [<JsonPropertyName("tape")>]
    Tape: int option
}

/// Response from the Polygon trades API
[<CLIMutable>]
type TradesResponse = {
    [<JsonPropertyName("results")>]
    Results: Trade[] option

    [<JsonPropertyName("status")>]
    Status: string

    [<JsonPropertyName("request_id")>]
    RequestId: string

    [<JsonPropertyName("next_url")>]
    NextUrl: string option
}

/// Result of downloading trades for a ticker/date
type TradesDownloadResult =
    | TradesDownloaded of ticker: string * date: DateTime * tradeCount: int
    | TradesSkipped of ticker: string * date: DateTime
    | TradesFailed of ticker: string * date: DateTime * error: string

let private baseUrl = "https://api.polygon.io/v3/trades"
let private maxRetries = 5
let private baseDelayMs = 200

let private jsonOptions =
    let options = JsonSerializerOptions()
    options.PropertyNameCaseInsensitive <- true
    options

let private jsonOptionsPretty =
    let options = JsonSerializerOptions()
    options.PropertyNameCaseInsensitive <- true
    options.WriteIndented <- true
    options

/// Generate output file path for trades data
let private getOutputPath (outputDir: string) (ticker: string) (date: DateTime) : string =
    let dateStr = date.ToString("yyyy-MM-dd")
    let dir = Path.Combine(outputDir, ticker)
    Directory.CreateDirectory(dir) |> ignore
    Path.Combine(dir, $"{dateStr}.json")

/// Download all trades for a single ticker and date (handles pagination)
let downloadTrades
    (httpClient: HttpClient)
    (apiKey: string)
    (ticker: string)
    (date: DateTime)
    (ct: CancellationToken)
    : Async<Result<Trade[], string>> =
    async {
        let dateStr = date.ToString("yyyy-MM-dd")
        let nextDateStr = date.AddDays(1.0).ToString("yyyy-MM-dd")

        // Build initial URL with timestamp range for the day
        let initialUrl = $"{baseUrl}/{ticker}?timestamp.gte={dateStr}&timestamp.lt={nextDateStr}&limit=50000&apiKey={apiKey}"

        let rec fetchPage (url: string) (accumulatedTrades: Trade list) attempt delay =
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
                            return! fetchPage url accumulatedTrades (attempt + 1) newDelay
                    elif response.StatusCode = HttpStatusCode.NotFound then
                        return Error $"No data found for {ticker} on {dateStr}"
                    else
                        response.EnsureSuccessStatusCode() |> ignore
                        let! content = response.Content.ReadAsStringAsync(ct) |> Async.AwaitTask
                        let parsed = JsonSerializer.Deserialize<TradesResponse>(content, jsonOptions)

                        let newTrades =
                            parsed.Results
                            |> Option.map Array.toList
                            |> Option.defaultValue []

                        let allTrades = accumulatedTrades @ newTrades

                        match parsed.NextUrl with
                        | Some nextUrl when not (String.IsNullOrEmpty nextUrl) ->
                            // Add API key to next_url if not present
                            let nextUrlWithKey =
                                if nextUrl.Contains("apiKey=") then nextUrl
                                else $"{nextUrl}&apiKey={apiKey}"
                            return! fetchPage nextUrlWithKey allTrades 0 baseDelayMs
                        | _ ->
                            return Ok (allTrades |> List.toArray)
                with
                | :? HttpRequestException as ex when attempt < maxRetries ->
                    do! Async.Sleep 500
                    return! fetchPage url accumulatedTrades (attempt + 1) delay
                | ex ->
                    return Error ex.Message
            }

        return! fetchPage initialUrl [] 0 baseDelayMs
    }

/// Download and save trades for a single ticker and date
let downloadAndSaveTrades
    (httpClient: HttpClient)
    (apiKey: string)
    (outputDir: string)
    (ticker: string)
    (date: DateTime)
    (prettyPrint: bool)
    (ct: CancellationToken)
    : Async<TradesDownloadResult> =
    async {
        let outputPath = getOutputPath outputDir ticker date

        if File.Exists outputPath then
            return TradesSkipped(ticker, date)
        else
            let! result = downloadTrades httpClient apiKey ticker date ct

            match result with
            | Error msg ->
                return TradesFailed(ticker, date, msg)
            | Ok trades ->
                let tradeCount = trades.Length

                // Save as JSON array
                let serializerOptions = if prettyPrint then jsonOptionsPretty else jsonOptions
                let json = JsonSerializer.Serialize(trades, serializerOptions)
                do! File.WriteAllTextAsync(outputPath, json, ct) |> Async.AwaitTask

                return TradesDownloaded(ticker, date, tradeCount)
    }

/// Progress callback type for trades downloads
type TradesProgressCallback = int -> int -> TradesDownloadResult -> unit

/// Download trades for multiple ticker/date pairs
let downloadTradesBatch
    (httpClient: HttpClient)
    (apiKey: string)
    (outputDir: string)
    (tickerDates: (string * DateTime) list)
    (prettyPrint: bool)
    (maxParallelism: int)
    (progress: TradesProgressCallback option)
    (ct: CancellationToken)
    : Async<TradesDownloadResult list> =
    async {
        let total = tickerDates.Length
        let completed = ref 0

        let reportProgress result =
            let c = Threading.Interlocked.Increment(completed)
            match progress with
            | Some callback -> callback c total result
            | None -> ()

        use semaphore = new SemaphoreSlim(maxParallelism, maxParallelism)

        let downloadWithSemaphore (ticker, date) =
            async {
                do! semaphore.WaitAsync(ct) |> Async.AwaitTask
                try
                    let! result = downloadAndSaveTrades httpClient apiKey outputDir ticker date prettyPrint ct
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
let consoleProgress (completed: int) (total: int) (result: TradesDownloadResult) : unit =
    let formatDate (d: DateTime) = d.ToString("yyyy-MM-dd")
    let status =
        match result with
        | TradesDownloaded(ticker, date, trades) ->
            $"Downloaded {ticker} {formatDate date} ({trades} trades)"
        | TradesSkipped(ticker, date) ->
            $"Skipped {ticker} {formatDate date} (exists)"
        | TradesFailed(ticker, date, error) ->
            $"Failed {ticker} {formatDate date}: {error}"

    printfn "[%d/%d] %s" completed total status
