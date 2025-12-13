module Spiral.Trading.CsvParsing

open System
open System.IO
open System.IO.Compression
open FSharp.Data

/// CSV schema for daily aggregate files from Massive
/// Sample: ticker,volume,open,close,high,low,window_start,transactions
type DailyAggCsv = CsvProvider<
    "ticker,volume,open,close,high,low,window_start,transactions
A,1547275,140.81,144.00,144.57,140.44,1733720400000000000,28350",
    Schema="ticker,volume (int64),open (float),close (float),high (float),low (float),window_start (int64),transactions (int64)">

/// Convert nanosecond Unix timestamp to DateTime
let private nanosToDateTime (nanos: int64) : DateTime =
    let ticksPerNano = 100L // 1 tick = 100 nanoseconds
    let unixEpoch = DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)
    unixEpoch.AddTicks(nanos / ticksPerNano)

/// Convert a CSV row to a DailyPrice record
let private rowToDailyPrice (row: DailyAggCsv.Row) : DailyPrice =
    {
        Ticker = row.Ticker
        Date = nanosToDateTime row.Window_start
        Open = row.Open
        High = row.High
        Low = row.Low
        Close = row.Close
        Volume = row.Volume
        Transactions = row.Transactions
    }

/// Parse a single gzipped CSV file
let parseGzipFile (filePath: string) : Result<DailyPrice array, string> =
    try
        use fileStream = File.OpenRead(filePath)
        use gzipStream = new GZipStream(fileStream, CompressionMode.Decompress)
        use reader = new StreamReader(gzipStream)

        let csv = DailyAggCsv.Load(reader)
        let prices = csv.Rows |> Seq.map rowToDailyPrice |> Seq.toArray
        Ok prices
    with ex ->
        Error $"Failed to parse {filePath}: {ex.Message}"

/// Parse a single uncompressed CSV file
let parseCsvFile (filePath: string) : Result<DailyPrice array, string> =
    try
        let csv = DailyAggCsv.Load(filePath)
        let prices = csv.Rows |> Seq.map rowToDailyPrice |> Seq.toArray
        Ok prices
    with ex ->
        Error $"Failed to parse {filePath}: {ex.Message}"

/// Result of parsing a file
type ParseResult = {
    FilePath: string
    FileName: string
    Date: DateTime option
    PriceCount: int
    Error: string option
}

/// Try to extract date from filename like "2024-12-09.csv.gz"
let private extractDateFromFileName (fileName: string) : DateTime option =
    let baseName = Path.GetFileNameWithoutExtension(Path.GetFileNameWithoutExtension(fileName))
    match DateTime.TryParse(baseName) with
    | true, date -> Some date
    | false, _ -> None

/// Parse a gzipped CSV file and return a ParseResult
let parseGzipFileWithResult (filePath: string) : ParseResult * DailyPrice array =
    let fileName = Path.GetFileName(filePath)
    let date = extractDateFromFileName fileName

    match parseGzipFile filePath with
    | Ok prices ->
        { FilePath = filePath
          FileName = fileName
          Date = date
          PriceCount = prices.Length
          Error = None }, prices
    | Error msg ->
        { FilePath = filePath
          FileName = fileName
          Date = date
          PriceCount = 0
          Error = Some msg }, [||]

/// Parse all gzipped CSV files in a directory
let parseDirectory (directoryPath: string) : (ParseResult * DailyPrice array) array =
    if not (Directory.Exists directoryPath) then
        [||]
    else
        Directory.GetFiles(directoryPath, "*.csv.gz")
        |> Array.map parseGzipFileWithResult

/// Parse all gzipped CSV files in a directory, returning only successful results
let parseDirectoryPrices (directoryPath: string) : DailyPrice array =
    parseDirectory directoryPath
    |> Array.collect snd

/// Parse directory with progress reporting
let parseDirectoryWithProgress
    (directoryPath: string)
    (progress: int -> int -> ParseResult -> unit)
    : (ParseResult * DailyPrice array) array =
    if not (Directory.Exists directoryPath) then
        [||]
    else
        let files = Directory.GetFiles(directoryPath, "*.csv.gz")
        let total = files.Length
        let mutable completed = 0

        files
        |> Array.map (fun filePath ->
            let result, prices = parseGzipFileWithResult filePath
            completed <- completed + 1
            progress completed total result
            result, prices)

/// Console progress reporter for parsing
let consoleParseProgress (completed: int) (total: int) (result: ParseResult) : unit =
    let status =
        match result.Error with
        | Some err -> sprintf "Failed (%s)" err
        | None -> sprintf "Parsed %d rows" result.PriceCount
    printfn "[%d/%d] %s: %s" completed total result.FileName status
