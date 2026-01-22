module Spiral.Trading.Simulation.DatasetGeneration

open System
open System.IO
open Parquet
open Parquet.Schema
open Parquet.Data
open Spiral.Trading.Simulation.EpisodeMCMC
open Spiral.Trading.Simulation.PriceGeneration

let sessionToInt (s: DaySession) : int =
    match s with
    | Morning -> 0
    | Mid -> 1
    | Close -> 2

let trendToInt (t: Trend) : int =
    match t with
    | StrongUptrend -> 0
    | MidUptrend -> 1
    | WeakUptrend -> 2
    | Consolidation -> 3
    | WeakDowntrend -> 4
    | MidDowntrend -> 5
    | StrongDowntrend -> 6

let generateDataset 
    (rng: Random) 
    (numDays: int) 
    (outputPath: string)
    (mcmcConfig: MCMC.Config)
    (sessionConfig: SessionLevel.Config)
    (trendConfig: TrendLevel.Config)
    (startPrice: float)
    : unit =
    
    let barsPerDay = 390 * 60
    let totalBars = numDays * barsPerDay
    let batchSize = 100 // days per batch
    
    printfn "Generating %d days (%d bars) in batches of %d days..." numDays totalBars batchSize
    
    let dir = Path.GetDirectoryName(outputPath)
    if not (String.IsNullOrEmpty(dir)) && not (Directory.Exists(dir)) then
        Directory.CreateDirectory(dir) |> ignore
    
    // Create schema
    let schema = ParquetSchema(
        DataField<int>("day_id"),
        DataField<int>("time"),
        DataField<float>("open"),
        DataField<float>("high"),
        DataField<float>("low"),
        DataField<float>("close"),
        DataField<int>("session"),
        DataField<int>("trend")
    )
    
    use stream = File.Create(outputPath)
    use writer = ParquetWriter.CreateAsync(schema, stream) |> Async.AwaitTask |> Async.RunSynchronously
    
    let mutable day = 0
    while day < numDays do
        let batchDays = min batchSize (numDays - day)
        let batchBars = batchDays * barsPerDay
        
        if day % 100 = 0 then
            printfn "  Day %d / %d" day numDays
        
        // Allocate batch arrays
        let dayIds = Array.zeroCreate<int> batchBars
        let times = Array.zeroCreate<int> batchBars
        let opens = Array.zeroCreate<float> batchBars
        let highs = Array.zeroCreate<float> batchBars
        let lows = Array.zeroCreate<float> batchBars
        let closes = Array.zeroCreate<float> batchBars
        let sessions = Array.zeroCreate<int> batchBars
        let trends = Array.zeroCreate<int> batchBars
        
        for d in 0 .. batchDays - 1 do
            let result = generateDay sessionConfig trendConfig mcmcConfig rng 390.0
            let bars = generateDayBars rng startPrice result
            
            let offset = d * barsPerDay
            for i in 0 .. bars.Length - 1 do
                let idx = offset + i
                dayIds.[idx] <- day + d
                times.[idx] <- int bars.[i].Time
                opens.[idx] <- bars.[i].Open
                highs.[idx] <- bars.[i].High
                lows.[idx] <- bars.[i].Low
                closes.[idx] <- bars.[i].Close
                sessions.[idx] <- sessionToInt bars.[i].Session
                trends.[idx] <- trendToInt bars.[i].Trend
        
        // Write batch as row group
        use rowGroup = writer.CreateRowGroup()
        rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[0], dayIds)) |> Async.AwaitTask |> Async.RunSynchronously
        rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[1], times)) |> Async.AwaitTask |> Async.RunSynchronously
        rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[2], opens)) |> Async.AwaitTask |> Async.RunSynchronously
        rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[3], highs)) |> Async.AwaitTask |> Async.RunSynchronously
        rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[4], lows)) |> Async.AwaitTask |> Async.RunSynchronously
        rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[5], closes)) |> Async.AwaitTask |> Async.RunSynchronously
        rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[6], sessions)) |> Async.AwaitTask |> Async.RunSynchronously
        rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[7], trends)) |> Async.AwaitTask |> Async.RunSynchronously
        
        day <- day + batchDays
    
    printfn "Done. Wrote %d bars to %s" totalBars outputPath
