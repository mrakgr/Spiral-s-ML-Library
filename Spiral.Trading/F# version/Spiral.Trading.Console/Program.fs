open System
open System.IO
open System.Net.Http
open System.Threading
open Argu
open Spiral.Trading
open Spiral.Trading.Config
open Spiral.Trading.S3Download
open Spiral.Trading.SplitDownload
open Spiral.Trading.CsvParsing
open Spiral.Trading.Database
open Spiral.Trading.Plotting

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

type ParseCsvArgs =
    | [<AltCommandLine("-d")>] Directory of string
    | [<AltCommandLine("-f")>] File of string

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Directory _ -> "Directory containing .csv.gz files (default: data/daily_aggregates)"
            | File _ -> "Single .csv.gz file to parse"

type IngestDataArgs =
    | [<AltCommandLine("-d")>] Database of string
    | [<AltCommandLine("-c")>] Csv_Dir of string
    | [<AltCommandLine("-s")>] Splits_File of string

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Database _ -> "SQLite database path (default: data/trading.db)"
            | Csv_Dir _ -> "Directory containing .csv.gz files (default: data/daily_aggregates)"
            | Splits_File _ -> "JSON file containing splits (default: data/splits.json)"

type PlotChartArgs =
    | [<AltCommandLine("-t")>] Ticker of string
    | [<AltCommandLine("-d")>] Database of string
    | [<AltCommandLine("-o")>] Output of string
    | [<AltCommandLine("-w")>] Width of int
    | [<AltCommandLine("-h")>] Height of int

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Ticker _ -> "Stock ticker symbol (required)"
            | Database _ -> "SQLite database path (default: data/trading.db)"
            | Output _ -> "Output HTML file path (default: data/{ticker}_chart.html)"
            | Width _ -> "Chart width in pixels (default: 1200)"
            | Height _ -> "Chart height in pixels (default: 900)"

type Arguments =
    | [<CliPrefix(CliPrefix.None)>] Download_Bulk of ParseResults<DownloadBulkArgs>
    | [<CliPrefix(CliPrefix.None)>] Download_Splits of ParseResults<DownloadSplitsArgs>
    | [<CliPrefix(CliPrefix.None)>] Parse_Csv of ParseResults<ParseCsvArgs>
    | [<CliPrefix(CliPrefix.None)>] Ingest_Data of ParseResults<IngestDataArgs>
    | [<CliPrefix(CliPrefix.None)>] Plot_Chart of ParseResults<PlotChartArgs>

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Download_Bulk _ -> "Download daily aggregate files from Massive S3"
            | Download_Splits _ -> "Download stock splits from Massive API"
            | Parse_Csv _ -> "Parse downloaded CSV files and display summary"
            | Ingest_Data _ -> "Ingest downloaded data into SQLite database"
            | Plot_Chart _ -> "Generate a candlestick chart for a ticker"

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

    let parallelism = args.GetResult(DownloadBulkArgs.Parallelism, defaultValue = 30)
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

let private handleParseCsv (args: ParseResults<ParseCsvArgs>) =
    match args.TryGetResult ParseCsvArgs.File with
    | Some filePath ->
        // Parse single file
        printfn "Parsing file: %s" filePath
        let result, prices = parseGzipFileWithResult filePath

        match result.Error with
        | Some err ->
            printfn "Error: %s" err
        | None ->
            printfn "Parsed %d price records" result.PriceCount

            // Show sample
            if not (Array.isEmpty prices) then
                printfn ""
                printfn "Sample records:"
                prices
                |> Array.take (min 5 prices.Length)
                |> Array.iter (fun p ->
                    printfn "  %s %s O:%.2f H:%.2f L:%.2f C:%.2f V:%d"
                        p.Ticker (formatDate p.Date) p.Open p.High p.Low p.Close p.Volume)

    | None ->
        // Parse directory
        let directory =
            args.TryGetResult ParseCsvArgs.Directory
            |> Option.defaultValue "data/daily_aggregates"

        printfn "Parsing directory: %s" (Path.GetFullPath directory)
        printfn ""

        let results = parseDirectoryWithProgress directory consoleParseProgress

        let totalFiles = results.Length
        let totalRows = results |> Array.sumBy (fun (r, _) -> r.PriceCount)
        let errors = results |> Array.filter (fun (r, _) -> r.Error.IsSome) |> Array.length

        printfn ""
        printfn "Parse complete: %d files, %d total rows, %d errors" totalFiles totalRows errors

let private handleIngestData (args: ParseResults<IngestDataArgs>) =
    ensureDataDir ()

    let dbPath =
        args.TryGetResult IngestDataArgs.Database
        |> Option.defaultValue "data/trading.db"

    let csvDir =
        args.TryGetResult IngestDataArgs.Csv_Dir
        |> Option.defaultValue "data/daily_aggregates"

    let splitsFile =
        args.TryGetResult IngestDataArgs.Splits_File
        |> Option.defaultValue "data/splits.json"

    printfn "Database: %s" (Path.GetFullPath dbPath)
    printfn "CSV directory: %s" (Path.GetFullPath csvDir)
    printfn "Splits file: %s" (Path.GetFullPath splitsFile)
    printfn ""

    use connection = openConnection dbPath
    initializeSchema connection

    // Get already processed files
    let processedFiles = getProcessedFiles connection
    printfn "Already processed: %d files" processedFiles.Count

    // Ingest daily prices from CSV files using DuckDB's native CSV reader with glob
    if Directory.Exists csvDir then
        let allFiles = Directory.GetFiles(csvDir, "*.csv.gz")
        printfn "Found %d CSV files" allFiles.Length

        let globPattern = Path.Combine(csvDir, "*.csv.gz")
        printfn "Bulk loading from: %s" globPattern
        
        let sw = System.Diagnostics.Stopwatch.StartNew()
        let countBefore = getDailyPriceCount connection
        let _ = ingestDailyPricesFromGlob connection globPattern
        let countAfter = getDailyPriceCount connection
        let totalPrices = countAfter - countBefore
        sw.Stop()

        // Mark all files as processed
        let fileNames = allFiles |> Array.map Path.GetFileName
        markFilesProcessed connection fileNames

        let rowsPerSec = if sw.Elapsed.TotalSeconds > 0.0 then float countAfter / sw.Elapsed.TotalSeconds else 0.0
        printfn "Ingested %d new daily prices (total: %d) in %.2fs (%.0f rows/sec)" totalPrices countAfter sw.Elapsed.TotalSeconds rowsPerSec
    else
        printfn "CSV directory not found: %s" csvDir

    // Ingest splits from JSON file
    if File.Exists splitsFile then
        let sw = System.Diagnostics.Stopwatch.StartNew()
        let json = File.ReadAllText splitsFile
        let splits = System.Text.Json.JsonSerializer.Deserialize<Split array>(json)
        let inserted = upsertSplits connection splits
        sw.Stop()
        printfn "Ingested %d splits in %.2fs" splits.Length sw.Elapsed.TotalSeconds
    else
        printfn "Splits file not found: %s" splitsFile

    printfn "Data ingestion complete."

    // Show summary
    printfn ""
    printfn "Database summary:"
    printfn "  Daily prices: %d" (getDailyPriceCount connection)
    printfn "  Splits: %d" (getSplitCount connection)

    match getDateRange connection with
    | Some (minDate, maxDate) ->
        printfn "  Date range: %s to %s" (formatDate minDate) (formatDate maxDate)
    | None ->
        printfn "  Date range: (no data)"

let private handlePlotChart (args: ParseResults<PlotChartArgs>) =
    let ticker =
        match args.TryGetResult PlotChartArgs.Ticker with
        | Some t -> t.ToUpperInvariant()
        | None -> failwith "Ticker is required. Use -t or --ticker to specify."

    let dbPath =
        args.TryGetResult PlotChartArgs.Database
        |> Option.defaultValue "data/trading.db"

    let outputPath =
        args.TryGetResult PlotChartArgs.Output
        |> Option.defaultValue $"data/{ticker}_chart.html"

    let width = args.GetResult(PlotChartArgs.Width, defaultValue = 1200)
    let height = args.GetResult(PlotChartArgs.Height, defaultValue = 900)

    printfn "Generating chart for %s" ticker
    printfn "Database: %s" (Path.GetFullPath dbPath)
    printfn "Output: %s" (Path.GetFullPath outputPath)

    Plotting.generateChart dbPath ticker outputPath width height

[<EntryPoint>]
let main argv =
    let parser = ArgumentParser.Create<Arguments>(programName = "Spiral.Trading")

    try
        let results = parser.ParseCommandLine(inputs = argv, raiseOnUsage = true)

        let configPath = Path.Combine(Environment.CurrentDirectory, "api_key.json")

        for result in results.GetAllResults() do
            match result with
            | Download_Bulk args ->
                let config = loadConfigOrFail configPath
                handleDownloadBulk config args
            | Download_Splits args ->
                let config = loadConfigOrFail configPath
                handleDownloadSplits config args
            | Parse_Csv args ->
                handleParseCsv args
            | Ingest_Data args ->
                handleIngestData args
            | Plot_Chart args ->
                handlePlotChart args

        0
    with
    | :? ArguParseException as ex ->
        printfn "%s" ex.Message
        1
    | ex ->
        printfn "Error: %s" ex.Message
        1
