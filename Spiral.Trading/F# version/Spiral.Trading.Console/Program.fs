open System
open System.IO
open System.Net.Http
open System.Threading
open Argu
open Spiral.Trading
open Spiral.Trading.Config
open Spiral.Trading.S3Download
open Spiral.Trading.SplitDownload

let private formatDate (d: DateTime) = d.ToString("yyyy-MM-dd")

type DownloadBulkArgs =
    | [<AltCommandLine("-s")>] Start_Date of string
    | [<AltCommandLine("-e")>] End_Date of string
    | [<AltCommandLine("-p")>] Parallelism of int

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Start_Date _ -> "Start date (yyyy-MM-dd). Default: 5 years ago"
            | End_Date _ -> "End date (yyyy-MM-dd). Default: today"
            | Parallelism _ -> "Max parallel downloads. Default: 8"

type DownloadSplitsArgs =
    | [<AltCommandLine("-s")>] Start_Date of string
    | [<AltCommandLine("-e")>] End_Date of string

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Start_Date _ -> "Start date (yyyy-MM-dd). Default: 5 years ago"
            | End_Date _ -> "End date (yyyy-MM-dd). Default: none (all future)"

type Arguments =
    | [<CliPrefix(CliPrefix.None)>] Download_Bulk of ParseResults<DownloadBulkArgs>
    | [<CliPrefix(CliPrefix.None)>] Download_Splits of ParseResults<DownloadSplitsArgs>

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Download_Bulk _ -> "Download daily aggregate files from Massive S3"
            | Download_Splits _ -> "Download stock splits from Massive API"

let private ensureDataDir () =
    Directory.CreateDirectory("data") |> ignore
    Directory.CreateDirectory("data/daily_aggregates") |> ignore

let private handleDownloadBulk (config: MassiveConfig) (args: ParseResults<DownloadBulkArgs>) =
    ensureDataDir ()

    let endDate =
        args.TryGetResult DownloadBulkArgs.End_Date
        |> Option.map DateTime.Parse
        |> Option.defaultValue DateTime.Now

    let startDate =
        args.TryGetResult DownloadBulkArgs.Start_Date
        |> Option.map DateTime.Parse
        |> Option.defaultValue (endDate.AddYears(-5))

    let parallelism = args.GetResult(DownloadBulkArgs.Parallelism, defaultValue = 8)
    let outputDir = "data/daily_aggregates"

    printfn "Downloading daily aggregates from %s to %s" (formatDate startDate) (formatDate endDate)
    printfn "Output directory: %s" (Path.GetFullPath outputDir)
    printfn "Parallelism: %d" parallelism

    use client = createS3Client config.S3AccessKey config.S3SecretKey
    use cts = new CancellationTokenSource()

    let results =
        downloadDailyAggregates client startDate endDate outputDir parallelism (Some consoleProgress) cts.Token
        |> Async.RunSynchronously

    let downloaded = results |> List.filter (function Downloaded _ -> true | _ -> false) |> List.length
    let skipped = results |> List.filter (function Skipped _ -> true | _ -> false) |> List.length
    let failed = results |> List.filter (function Failed _ -> true | _ -> false) |> List.length

    printfn ""
    printfn "Download complete: %d downloaded, %d skipped, %d failed" downloaded skipped failed

let private handleDownloadSplits (config: MassiveConfig) (args: ParseResults<DownloadSplitsArgs>) =
    ensureDataDir ()

    let startDate =
        args.TryGetResult DownloadSplitsArgs.Start_Date
        |> Option.map DateTime.Parse
        |> Option.defaultValue (DateTime.Now.AddYears(-5))

    let endDate =
        args.TryGetResult DownloadSplitsArgs.End_Date
        |> Option.map DateTime.Parse

    let endDateStr =
        match endDate with
        | Some d -> sprintf " to %s" (formatDate d)
        | None -> ""

    printfn "Downloading splits from %s%s" (formatDate startDate) endDateStr

    use httpClient = new HttpClient()
    use cts = new CancellationTokenSource()

    let result =
        downloadSplitsWithConsoleProgress httpClient config.ApiKey startDate endDate cts.Token
        |> Async.RunSynchronously

    match result with
    | Ok splits ->
        printfn ""
        printfn "Downloaded %d splits" splits.Length

        // Save to JSON for now (we'll add DB storage later)
        let outputPath = "data/splits.json"
        let options = System.Text.Json.JsonSerializerOptions(WriteIndented = true)
        let json = System.Text.Json.JsonSerializer.Serialize(splits, options)
        File.WriteAllText(outputPath, json)
        printfn "Saved splits to %s" (Path.GetFullPath outputPath)

    | Error msg ->
        printfn "Error downloading splits: %s" msg

[<EntryPoint>]
let main argv =
    let parser = ArgumentParser.Create<Arguments>(programName = "Spiral.Trading")

    try
        let results = parser.ParseCommandLine(inputs = argv, raiseOnUsage = true)

        let configPath = Path.Combine(Environment.CurrentDirectory, "api_key.json")
        let config = loadConfigOrFail configPath

        for result in results.GetAllResults() do
            match result with
            | Download_Bulk args -> handleDownloadBulk config args
            | Download_Splits args -> handleDownloadSplits config args

        0
    with
    | :? ArguParseException as ex ->
        printfn "%s" ex.Message
        1
    | ex ->
        printfn "Error: %s" ex.Message
        1
