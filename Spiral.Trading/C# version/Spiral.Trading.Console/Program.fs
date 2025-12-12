open System
open System.IO
open Argu
open Spiral.Trading.Config
open Spiral.Trading.Data
open Spiral.Trading.Storage
open Spiral.Trading.Plotting
open Spiral.Trading.Analysis
open Spiral.Trading.Analysis.Models
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

type ScreenArgs =
    | [<Mandatory>] Year of int
    | [<Mandatory>] Month of int
    | Min_Relative_Volume of float
    | Min_Price_Change of float
    | Min_Days of int
    | Require_Both
    | Export_Csv of string
    | Max_Results of int

    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Year _ -> "Year to screen (e.g. 2024)"
            | Month _ -> "Month to screen (1-12)"
            | Min_Relative_Volume _ -> "Minimum relative volume multiplier (default: 3.0)"
            | Min_Price_Change _ -> "Minimum daily price change percent (default: 5.0)"
            | Min_Days _ -> "Minimum trading days required for ticker (default: 10)"
            | Require_Both -> "Require BOTH volume AND price criteria (default: OR)"
            | Export_Csv _ -> "Export results to CSV file path"
            | Max_Results _ -> "Maximum results to display (default: 50)"

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
    | Clear_Splits
    | Plot of ParseResults<PlotArgs>
    | Screen of ParseResults<ScreenArgs>

    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Download_Bulk _ -> "Download daily aggregate files from S3 to disk."
            | Ingest_Data -> "Ingest downloaded CSV files into SQLite database."
            | Download_Splits -> "Download stock splits from Polygon API to Database."
            | Clear_Splits -> "Clear all splits from the database."
            | Plot _ -> "Generate a candlestick chart for a ticker."
            | Screen _ -> "Screen for 'Stocks In Play' based on volume and price movement."

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

            | Clear_Splits ->
                ensureDb ()
                let ingestor = DataIngestor(dbPath)
                ingestor.ClearSplitsAsync().GetAwaiter().GetResult()

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

            | Screen screenArgs ->
                ensureDb ()
                let year = screenArgs.GetResult Year
                let month = screenArgs.GetResult Month

                let criteria = ScreeningCriteria(
                    MinRelativeVolume = screenArgs.GetResult(Min_Relative_Volume, defaultValue = 3.0),
                    MinPriceChangePercent = screenArgs.GetResult(Min_Price_Change, defaultValue = 5.0),
                    MinimumDaysInMonth = screenArgs.GetResult(Min_Days, defaultValue = 10),
                    RequireBothCriteria = screenArgs.Contains Require_Both
                )

                printfn "Screening for Stocks In Play: %d-%02d" year month
                printfn "Criteria: %.1fx relative volume, %.1f%% price change, %s logic"
                    criteria.MinRelativeVolume
                    criteria.MinPriceChangePercent
                    (if criteria.RequireBothCriteria then "AND" else "OR")

                let service = StockScreeningService(dbPath)
                let results = service.ScreenStocksInPlayAsync(year, month, criteria).GetAwaiter().GetResult()

                let maxResults = screenArgs.GetResult(Max_Results, defaultValue = 50)
                service.PrintResults(results, maxResults)

                match screenArgs.TryGetResult Export_Csv with
                | Some path -> service.ExportToCsv(results, path)
                | None -> ()

        0
    with e ->
        printfn "%s" e.Message
        1
