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
    
    { DayId = dayId; DayIds = dayIds; Times = times; Opens = opens
      Highs = highs; Lows = lows; Closes = closes; Sessions = sessions; Trends = trends }

let writerTask 
    (schema: ParquetSchema)
    (outputPath: string)
    (numDays: int)
    (channel: Channel<DayData>) 
    = task {
    use stream = File.Create(outputPath)
    let! writer = ParquetWriter.CreateAsync(schema, stream)
    use writer = writer
    
    let reader = channel.Reader
    let mutable daysWritten = 0
    
    let mutable hasMore = true
    while hasMore do
        let! canRead = reader.WaitToReadAsync()
        if canRead then
            let mutable data = Unchecked.defaultof<DayData>
            while reader.TryRead(&data) do
                use rowGroup = writer.CreateRowGroup()
                do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[0], data.DayIds))
                do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[1], data.Times))
                do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[2], data.Opens))
                do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[3], data.Highs))
                do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[4], data.Lows))
                do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[5], data.Closes))
                do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[6], data.Sessions))
                do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[7], data.Trends))
                
                daysWritten <- daysWritten + 1
                if daysWritten % 500 = 0 then
                    printfn "  Written %d / %d days" daysWritten numDays
        else
            hasMore <- false
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
        DataField<int>("trend")
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
