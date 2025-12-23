module Spiral.Trading.QuotesDownload

open System
open System.IO
open System.Net
open System.Net.Http
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading

/// A single quote (NBBO) from the Polygon API
[<CLIMutable>]
type Quote = {
    [<JsonPropertyName("sip_timestamp")>]
    SipTimestamp: int64

    [<JsonPropertyName("participant_timestamp")>]
    ParticipantTimestamp: int64

    [<JsonPropertyName("sequence_number")>]
    SequenceNumber: int64

    [<JsonPropertyName("bid_price")>]
    BidPrice: float

    [<JsonPropertyName("bid_size")>]
    BidSize: float

    [<JsonPropertyName("bid_exchange")>]
    BidExchange: int

    [<JsonPropertyName("ask_price")>]
    AskPrice: float

    [<JsonPropertyName("ask_size")>]
    AskSize: float

    [<JsonPropertyName("ask_exchange")>]
    AskExchange: int

    [<JsonPropertyName("conditions")>]
    Conditions: int[] option

    [<JsonPropertyName("indicators")>]
    Indicators: int[] option

    [<JsonPropertyName("tape")>]
    Tape: int option
}

/// Response from the Polygon quotes API
[<CLIMutable>]
type QuotesResponse = {
    [<JsonPropertyName("results")>]
    Results: Quote[] option

    [<JsonPropertyName("status")>]
    Status: string

    [<JsonPropertyName("request_id")>]
    RequestId: string

    [<JsonPropertyName("next_url")>]
    NextUrl: string option
}

/// Result of downloading quotes for a ticker/date
type QuotesDownloadResult =
    | QuotesDownloaded of ticker: string * date: DateTime * quoteCount: int
    | QuotesSkipped of ticker: string * date: DateTime
    | QuotesFailed of ticker: string * date: DateTime * error: string

let private baseUrl = "https://api.polygon.io/v3/quotes"
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

/// Generate output file path for quotes data
let private getOutputPath (outputDir: string) (ticker: string) (date: DateTime) : string =
    let dateStr = date.ToString("yyyy-MM-dd")
    let dir = Path.Combine(outputDir, ticker)
    Directory.CreateDirectory(dir) |> ignore
    Path.Combine(dir, $"{dateStr}.json")

/// Download all quotes for a single ticker and date (handles pagination)
let downloadQuotes
    (httpClient: HttpClient)
    (apiKey: string)
    (ticker: string)
    (date: DateTime)
    (ct: CancellationToken)
    : Async<Result<Quote[], string>> =
    async {
        let dateStr = date.ToString("yyyy-MM-dd")
        let nextDateStr = date.AddDays(1.0).ToString("yyyy-MM-dd")

        let initialUrl = $"{baseUrl}/{ticker}?timestamp.gte={dateStr}&timestamp.lt={nextDateStr}&limit=50000&apiKey={apiKey}"

        let rec fetchPage (url: string) (accumulatedQuotes: Quote list) attempt delay =
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
                            return! fetchPage url accumulatedQuotes (attempt + 1) newDelay
                    elif response.StatusCode = HttpStatusCode.NotFound then
                        return Error $"No data found for {ticker} on {dateStr}"
                    else
                        response.EnsureSuccessStatusCode() |> ignore
                        let! content = response.Content.ReadAsStringAsync(ct) |> Async.AwaitTask
                        let parsed = JsonSerializer.Deserialize<QuotesResponse>(content, jsonOptions)

                        let newQuotes =
                            parsed.Results
                            |> Option.map Array.toList
                            |> Option.defaultValue []

                        let allQuotes = accumulatedQuotes @ newQuotes

                        match parsed.NextUrl with
                        | Some nextUrl when not (String.IsNullOrEmpty nextUrl) ->
                            let nextUrlWithKey =
                                if nextUrl.Contains("apiKey=") then nextUrl
                                else $"{nextUrl}&apiKey={apiKey}"
                            return! fetchPage nextUrlWithKey allQuotes 0 baseDelayMs
                        | _ ->
                            return Ok (allQuotes |> List.toArray)
                with
                | :? HttpRequestException as ex when attempt < maxRetries ->
                    do! Async.Sleep 500
                    return! fetchPage url accumulatedQuotes (attempt + 1) delay
                | ex ->
                    return Error ex.Message
            }

        return! fetchPage initialUrl [] 0 baseDelayMs
    }

/// Download and save quotes for a single ticker and date
let downloadAndSaveQuotes
    (httpClient: HttpClient)
    (apiKey: string)
    (outputDir: string)
    (ticker: string)
    (date: DateTime)
    (prettyPrint: bool)
    (ct: CancellationToken)
    : Async<QuotesDownloadResult> =
    async {
        let outputPath = getOutputPath outputDir ticker date

        if File.Exists outputPath then
            return QuotesSkipped(ticker, date)
        else
            let! result = downloadQuotes httpClient apiKey ticker date ct

            match result with
            | Error msg ->
                return QuotesFailed(ticker, date, msg)
            | Ok quotes ->
                let quoteCount = quotes.Length

                let serializerOptions = if prettyPrint then jsonOptionsPretty else jsonOptions
                let json = JsonSerializer.Serialize(quotes, serializerOptions)
                do! File.WriteAllTextAsync(outputPath, json, ct) |> Async.AwaitTask

                return QuotesDownloaded(ticker, date, quoteCount)
    }

/// Progress callback type for quotes downloads
type QuotesProgressCallback = int -> int -> QuotesDownloadResult -> unit

/// Download quotes for multiple ticker/date pairs
let downloadQuotesBatch
    (httpClient: HttpClient)
    (apiKey: string)
    (outputDir: string)
    (tickerDates: (string * DateTime) list)
    (prettyPrint: bool)
    (maxParallelism: int)
    (progress: QuotesProgressCallback option)
    (ct: CancellationToken)
    : Async<QuotesDownloadResult list> =
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
                    let! result = downloadAndSaveQuotes httpClient apiKey outputDir ticker date prettyPrint ct
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
let consoleProgress (completed: int) (total: int) (result: QuotesDownloadResult) : unit =
    let formatDate (d: DateTime) = d.ToString("yyyy-MM-dd")
    let status =
        match result with
        | QuotesDownloaded(ticker, date, quotes) ->
            $"Downloaded {ticker} {formatDate date} ({quotes} quotes)"
        | QuotesSkipped(ticker, date) ->
            $"Skipped {ticker} {formatDate date} (exists)"
        | QuotesFailed(ticker, date, error) ->
            $"Failed {ticker} {formatDate date}: {error}"

    printfn "[%d/%d] %s" completed total status
