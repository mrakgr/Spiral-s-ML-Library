module Spiral.Trading.Simulation.DatasetGeneration

open System
open System.IO
open System.Threading.Tasks
open System.Threading.Channels
open Parquet
open Parquet.Schema
open Parquet.Data
open Spiral.Trading.Simulation.EpisodeMCMC
open Spiral.Trading.Simulation.PriceGeneration
open FSharp.Control

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

let computeDeltas (opens: float[]) (highs: float[]) (lows: float[]) (closes: float[]) (periodSeconds: int) =
    let n = opens.Length
    let deltaOpens = Array.zeroCreate<float> n
    let deltaHighs = Array.zeroCreate<float> n
    let deltaLows = Array.zeroCreate<float> n
    let deltaCloses = Array.zeroCreate<float> n
    
    let mutable periodOpen = 0.0
    let mutable runningHigh = 0.0
    let mutable runningLow = 0.0
    
    for i in 0 .. n - 1 do
        let posInPeriod = i % periodSeconds
        if posInPeriod = 0 then
            periodOpen <- opens.[i]
            runningHigh <- highs.[i]
            runningLow <- lows.[i]
        else
            if highs.[i] > runningHigh then runningHigh <- highs.[i]
            if lows.[i] < runningLow then runningLow <- lows.[i]
        
        let scale = 100.0 / periodOpen
        deltaOpens.[i] <- 0.0
        deltaHighs.[i] <- (runningHigh - periodOpen) * scale
        deltaLows.[i] <- (runningLow - periodOpen) * scale
        deltaCloses.[i] <- (closes.[i] - periodOpen) * scale
    
    (deltaOpens, deltaHighs, deltaLows, deltaCloses)

let computePartials (opens: float[]) (highs: float[]) (lows: float[]) (closes: float[]) (periodSeconds: int) =
    let n = opens.Length
    let partialOpens = Array.zeroCreate<float> n
    let partialHighs = Array.zeroCreate<float> n
    let partialLows = Array.zeroCreate<float> n
    let partialCloses = Array.zeroCreate<float> n
    
    let mutable periodOpen = 0.0
    let mutable runningHigh = 0.0
    let mutable runningLow = 0.0
    
    for i in 0 .. n - 1 do
        let posInPeriod = i % periodSeconds
        if posInPeriod = 0 then
            periodOpen <- opens.[i]
            runningHigh <- highs.[i]
            runningLow <- lows.[i]
        else
            if highs.[i] > runningHigh then runningHigh <- highs.[i]
            if lows.[i] < runningLow then runningLow <- lows.[i]
        
        partialOpens.[i] <- periodOpen
        partialHighs.[i] <- runningHigh
        partialLows.[i] <- runningLow
        partialCloses.[i] <- closes.[i]
    
    (partialOpens, partialHighs, partialLows, partialCloses)

type DayData = {
    DayId: int
    DayIds: int[]
    Times: int[]
    Opens: float[]
    Highs: float[]
    Lows: float[]
    Closes: float[]
    Sessions: int[]
    Trends: int[]
    // 1-second deltas (% from bar open)
    DeltaHigh1s: float[]
    DeltaLow1s: float[]
    DeltaClose1s: float[]
    // 1-minute deltas (% from period open)
    DeltaHigh1m: float[]
    DeltaLow1m: float[]
    DeltaClose1m: float[]
    // 5-minute deltas (% from period open)
    DeltaHigh5m: float[]
    DeltaLow5m: float[]
    DeltaClose5m: float[]
}

let generateSingleDay 
    (dayId: int)
    (seed: int)
    (mcmcConfig: MCMC.Config)
    (sessionConfig: SessionLevel.Config)
    (trendConfig: TrendLevel.Config)
    (startPrice: float)
    : DayData =
    
    let barsPerDay = 390 * 60
    let rng = Random(seed)
    
    let result = generateDay sessionConfig trendConfig mcmcConfig rng 390.0
    let bars = generateDayBars rng startPrice result
    
    let dayIds = Array.create barsPerDay dayId
    let times = Array.zeroCreate<int> barsPerDay
    let opens = Array.zeroCreate<float> barsPerDay
    let highs = Array.zeroCreate<float> barsPerDay
    let lows = Array.zeroCreate<float> barsPerDay
    let closes = Array.zeroCreate<float> barsPerDay
    let sessions = Array.zeroCreate<int> barsPerDay
    let trends = Array.zeroCreate<int> barsPerDay
    
    for i in 0 .. bars.Length - 1 do
        times.[i] <- int bars.[i].Time
        opens.[i] <- bars.[i].Open
        highs.[i] <- bars.[i].High
        lows.[i] <- bars.[i].Low
        closes.[i] <- bars.[i].Close
        sessions.[i] <- sessionToInt bars.[i].Session
        trends.[i] <- trendToInt bars.[i].Trend
    
    // Compute 1-second deltas (period = 1)
    let (_, deltaHigh1s, deltaLow1s, deltaClose1s) = 
        computeDeltas opens highs lows closes 1
    
    // Compute 1-minute deltas (60 seconds per bar)
    let (_, deltaHigh1m, deltaLow1m, deltaClose1m) = 
        computeDeltas opens highs lows closes 60
    
    // Compute 5-minute deltas (300 seconds per bar)
    let (_, deltaHigh5m, deltaLow5m, deltaClose5m) = 
        computeDeltas opens highs lows closes 300
    
    { DayId = dayId; DayIds = dayIds; Times = times; Opens = opens
      Highs = highs; Lows = lows; Closes = closes; Sessions = sessions; Trends = trends
      DeltaHigh1s = deltaHigh1s; DeltaLow1s = deltaLow1s; DeltaClose1s = deltaClose1s
      DeltaHigh1m = deltaHigh1m; DeltaLow1m = deltaLow1m; DeltaClose1m = deltaClose1m
      DeltaHigh5m = deltaHigh5m; DeltaLow5m = deltaLow5m; DeltaClose5m = deltaClose5m }

let writerTask 
    (schema: ParquetSchema)
    (outputPath: string)
    (numDays: int)
    (channel: Channel<DayData>) 
    = task {
    use stream = File.Create(outputPath)
    let! writer = ParquetWriter.CreateAsync(schema, stream)
    use writer = writer
    
    let mutable daysWritten = 0
    
    for data in channel.Reader.ReadAllAsync() do
        use rowGroup = writer.CreateRowGroup()
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[0], data.DayIds))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[1], data.Times))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[2], data.Opens))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[3], data.Highs))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[4], data.Lows))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[5], data.Closes))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[6], data.Sessions))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[7], data.Trends))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[8], data.DeltaHigh1s))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[9], data.DeltaLow1s))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[10], data.DeltaClose1s))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[11], data.DeltaHigh1m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[12], data.DeltaLow1m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[13], data.DeltaClose1m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[14], data.DeltaHigh5m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[15], data.DeltaLow5m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[16], data.DeltaClose5m))
        
        daysWritten <- daysWritten + 1
        if daysWritten % 500 = 0 then
            printfn "  Written %d / %d days" daysWritten numDays
}

let generatorTask
    (workerId: int)
    (numWorkers: int)
    (numDays: int)
    (baseSeed: int)
    (mcmcConfig: MCMC.Config)
    (sessionConfig: SessionLevel.Config)
    (trendConfig: TrendLevel.Config)
    (startPrice: float)
    (channel: Channel<DayData>)
    = task {
    let writer = channel.Writer
    let mutable dayId = workerId
    while dayId < numDays do
        let seed = baseSeed + dayId
        let data = generateSingleDay dayId seed mcmcConfig sessionConfig trendConfig startPrice
        do! writer.WriteAsync(data)
        dayId <- dayId + numWorkers
}

let generateDataset 
    (baseSeed: int) 
    (numDays: int) 
    (outputPath: string)
    (mcmcConfig: MCMC.Config)
    (sessionConfig: SessionLevel.Config)
    (trendConfig: TrendLevel.Config)
    (startPrice: float)
    : unit =
    
    let barsPerDay = 390 * 60
    let totalBars = numDays * barsPerDay
    let numWorkers = Environment.ProcessorCount
    
    printfn "Generating %d days (%d bars) with %d workers..." numDays totalBars numWorkers
    
    let dir = Path.GetDirectoryName(outputPath)
    if not (String.IsNullOrEmpty(dir)) && not (Directory.Exists(dir)) then
        Directory.CreateDirectory(dir) |> ignore
    
    let schema = ParquetSchema(
        DataField<int>("day_id"),
        DataField<int>("time"),
        DataField<float>("open"),
        DataField<float>("high"),
        DataField<float>("low"),
        DataField<float>("close"),
        DataField<int>("session"),
        DataField<int>("trend"),
        DataField<float>("delta_high_1s"),
        DataField<float>("delta_low_1s"),
        DataField<float>("delta_close_1s"),
        DataField<float>("delta_high_1m"),
        DataField<float>("delta_low_1m"),
        DataField<float>("delta_close_1m"),
        DataField<float>("delta_high_5m"),
        DataField<float>("delta_low_5m"),
        DataField<float>("delta_close_5m")
    )
    
    let channel = Channel.CreateBounded<DayData>(BoundedChannelOptions(numWorkers * 2))
    
    let writer = writerTask schema outputPath numDays channel
    
    let generators = 
        [| for w in 0 .. numWorkers - 1 ->
            Task.Run(Func<Task>(fun () -> 
                generatorTask w numWorkers numDays baseSeed mcmcConfig sessionConfig trendConfig startPrice channel)) |]
    
    Task.WhenAll(generators).ContinueWith(Action<Task>(fun _ -> channel.Writer.Complete())).Wait()
    writer.Wait()
    
    printfn "Done. Wrote %d bars to %s" totalBars outputPath
