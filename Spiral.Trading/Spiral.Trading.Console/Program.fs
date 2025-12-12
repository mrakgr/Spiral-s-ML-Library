open System
open System.IO
open Argu
open Spiral.Trading.Config
open Spiral.Trading.Data
open Spiral.Trading.Storage
open Spiral.Trading.Plotting
open Microsoft.EntityFrameworkCore

type PlotArgs =
    | [<Mandatory>] Ticker of string
    | Width of int
    | Height of int

    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Ticker _ -> "Ticker symbol to plot (e.g. NVDA)"
            | Width _ -> "Chart width in pixels (default: 1600)"
            | Height _ -> "Chart height in pixels (default: 900)"

type BulkDownloadArgs =
    | Start_Date of string
    | End_Date of string

    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Start_Date _ -> "Start date (yyyy-MM-dd)"
            | End_Date _ -> "End date (yyyy-MM-dd)"

type Arguments =
    | Download_Bulk of ParseResults<BulkDownloadArgs>
    | Ingest_Data
    | Download_Splits
    | Plot of ParseResults<PlotArgs>

    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Download_Bulk _ -> "Download daily aggregate files from S3 to disk."
            | Ingest_Data -> "Ingest downloaded CSV files into SQLite database."
            | Download_Splits -> "Download stock splits from Polygon API to Database."
            | Plot _ -> "Generate a candlestick chart for a ticker."

[<EntryPoint>]
let main argv =
    let parser =
        ArgumentParser.Create<Arguments>(programName = "Spiral.Trading.Console")

    try
        let results = parser.ParseCommandLine(inputs = argv, raiseOnUsage = true)

        // Common setup
        let apiKeyPath = Path.Combine(Environment.CurrentDirectory, "api_key.json")
        let dbPath = "data/trading.db"
        Directory.CreateDirectory("data") |> ignore

        let ensureDb () =
            use context = new TradingDbContext(dbPath)
            context.Database.EnsureCreatedAsync().GetAwaiter().GetResult() |> ignore

        let loadKeys () =
            if not (File.Exists apiKeyPath) then
                failwithf "API key file not found at %s" apiKeyPath

            ConfigLoader.LoadKeys(apiKeyPath)

        for result in results.GetAllResults() do
            match result with
            | Ingest_Data ->
                ensureDb ()
                let ingestor = DataIngestor(dbPath)
                let dataDir = "data/daily_aggregates"

                if not (Directory.Exists dataDir) then
                    printfn "Data directory not found: %s" dataDir
                else
                    ingestor.IngestDailyAggregatesAsync(dataDir).GetAwaiter().GetResult()

            | Download_Splits ->
                let struct (apiKey, _, _) = loadKeys ()
                ensureDb ()
                let ingestor = DataIngestor(dbPath)
                ingestor.SyncSplitsFromPolygonBulkAsync(apiKey).GetAwaiter().GetResult()

            | Download_Bulk args ->
                let struct (_, s3Access, s3Secret) = loadKeys ()
                ensureDb ()
                let downloader = PolygonS3DataDownloader(s3Access, s3Secret)

                let endDate =
                    match args.TryGetResult End_Date with
                    | Some d -> DateTime.Parse d
                    | None -> DateTime.Now

                let startDate =
                    match args.TryGetResult Start_Date with
                    | Some d -> DateTime.Parse d
                    | None -> endDate.AddYears(-5)

                let outputDir = "data/daily_aggregates"
                printfn "Output Directory: %s" outputDir

                downloader.DownloadDailyAggregatesAsync(startDate, endDate, outputDir, 8).GetAwaiter().GetResult()

            | Plot plotArgs ->
                ensureDb ()
                let ticker = plotArgs.GetResult Ticker
                let width = plotArgs.GetResult(Width, defaultValue = 1600)
                let height = plotArgs.GetResult(Height, defaultValue = 900)
                let outputPath = sprintf "%s_chart.html" ticker

                printfn "Generating chart for %s..." ticker
                StockChartService.GenerateChart(Path.GetFullPath dbPath, ticker, outputPath, width, height)
                printfn "Chart saved to %s" (Path.GetFullPath outputPath)

        0
    with e ->
        printfn "%s" e.Message
        1
