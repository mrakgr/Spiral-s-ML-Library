module Spiral.Trading.SplitDownload

open System
open System.Net
open System.Net.Http
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading

let private jsonOptions =
    JsonFSharpOptions.Default()
        .WithSkippableOptionFields()
        .ToJsonSerializerOptions()

/// JSON response types for the Massive splits API
[<CLIMutable>]
type private SplitResult = {
    [<JsonPropertyName("ticker")>]
    Ticker: string

    [<JsonPropertyName("execution_date")>]
    ExecutionDate: string

    [<JsonPropertyName("split_from")>]
    SplitFrom: float

    [<JsonPropertyName("split_to")>]
    SplitTo: float
}

[<CLIMutable>]
type private SplitsResponse = {
    [<JsonPropertyName("results")>]
    Results: SplitResult[] option

    [<JsonPropertyName("next_url")>]
    NextUrl: string option
}

let private baseUrl = "https://api.polygon.io/v3/reference/splits"
let private maxRetries = 5
let private baseDelayMs = 200

/// Download all splits from the Massive API for a date range
let downloadSplits
    (httpClient: HttpClient)
    (apiKey: string)
    (startDate: DateTime)
    (endDate: DateTime option)
    (progress: (int -> int -> unit) option)
    (ct: CancellationToken)
    : Async<Result<Split list, string>> =
    async {
        let startDateStr = startDate.ToString("yyyy-MM-dd")
        let endDateStr = endDate |> Option.map (fun d -> d.ToString("yyyy-MM-dd"))

        let allSplits = ResizeArray<Split>()

        let buildUrl (nextUrl: string option) =
            match nextUrl with
            | Some n when n.Contains("apiKey=") -> n
            | Some n -> $"{n}&apiKey={apiKey}"
            | None ->
                let baseQuery = $"{baseUrl}?execution_date.gte={startDateStr}"
                let withEnd =
                    match endDateStr with
                    | Some e -> $"{baseQuery}&execution_date.lte={e}"
                    | None -> baseQuery
                $"{withEnd}&limit=1000&apiKey={apiKey}"

        let rec fetchPage (nextUrl: string option) (pageCount: int) : Async<Result<unit, string>> =
            async {
                if ct.IsCancellationRequested then
                    return Ok ()
                else
                    let url = buildUrl nextUrl

                    let rec retry attempt delay =
                        async {
                            try
                                let! response = httpClient.GetAsync(url, ct) |> Async.AwaitTask

                                if response.StatusCode = HttpStatusCode.TooManyRequests then
                                    if attempt >= maxRetries then
                                        return Error "Rate limited: max retries exceeded"
                                    else
                                        let jitter = Random.Shared.Next(10, 100)
                                        let currentDelay = int (float delay) + jitter
                                        do! Async.Sleep currentDelay
                                        let newDelay = int (float delay * 1.5)
                                        return! retry (attempt + 1) newDelay
                                else
                                    response.EnsureSuccessStatusCode() |> ignore
                                    let! content = response.Content.ReadAsStringAsync(ct) |> Async.AwaitTask
                                    let parsed = JsonSerializer.Deserialize<SplitsResponse>(content, jsonOptions)
                                    return Ok parsed
                            with ex when attempt < maxRetries ->
                                do! Async.Sleep 500
                                return! retry (attempt + 1) delay
                        }

                    let! result = retry 0 baseDelayMs

                    match result with
                    | Error msg -> return Error msg
                    | Ok response ->
                        match response.Results with
                        | Some results ->
                            for r in results do
                                if r.SplitFrom <> 0.0 && not (String.IsNullOrWhiteSpace r.Ticker) then
                                    allSplits.Add {
                                        Ticker = r.Ticker
                                        ExecutionDate = DateTime.Parse(r.ExecutionDate)
                                        SplitFrom = r.SplitFrom
                                        SplitTo = r.SplitTo
                                        SplitRatio = r.SplitTo / r.SplitFrom
                                    }
                        | None -> ()

                        let newPageCount = pageCount + 1

                        match progress with
                        | Some report -> report newPageCount allSplits.Count
                        | None -> ()

                        match response.NextUrl with
                        | Some n when not (String.IsNullOrWhiteSpace n) ->
                            return! fetchPage (Some n) newPageCount
                        | _ ->
                            return Ok ()
            }

        let! result = fetchPage None 0

        match result with
        | Error msg -> return Error msg
        | Ok () -> return Ok (allSplits |> Seq.toList)
    }

/// Download splits with console progress reporting
let downloadSplitsWithConsoleProgress
    (httpClient: HttpClient)
    (apiKey: string)
    (startDate: DateTime)
    (endDate: DateTime option)
    (ct: CancellationToken)
    : Async<Result<Split list, string>> =
    let progress page total =
        printfn "Page %d: Downloaded splits (Total: %d)" page total

    downloadSplits httpClient apiKey startDate endDate (Some progress) ct
