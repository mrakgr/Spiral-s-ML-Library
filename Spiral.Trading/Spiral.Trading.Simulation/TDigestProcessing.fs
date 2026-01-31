module Spiral.Trading.Simulation.TDigestProcessing

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open System.Threading.Channels
open Parquet
open Parquet.Schema
open Parquet.Data
open TDigest
open FSharp.Control

type TDigests = {
    PriceDeltas: MergingDigest
}

type private DeltaArrays = {
    DeltaHigh1s: float[]
    DeltaLow1s: float[]
    DeltaClose1s: float[]
    DeltaHigh1m: float[]
    DeltaLow1m: float[]
    DeltaClose1m: float[]
    DeltaHigh5m: float[]
    DeltaLow5m: float[]
    DeltaClose5m: float[]
}

type private RowGroupData = {
    Index: int
    DayIds: int[]
    Times: int[]
    Opens: float[]
    Highs: float[]
    Lows: float[]
    Closes: float[]
    Sessions: int[]
    Trends: int[]
    Deltas: DeltaArrays
}

let createTDigests (compression: float) =
    { PriceDeltas = MergingDigest(compression) }

let defaultCompression = 4096.0 // 2^12

let private readDeltaArrays (rowGroupReader: ParquetRowGroupReader) (schema: ParquetSchema) = task {
    let! deltaHigh1sCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[8])
    let! deltaLow1sCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[9])
    let! deltaClose1sCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[10])
    let! deltaHigh1mCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[11])
    let! deltaLow1mCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[12])
    let! deltaClose1mCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[13])
    let! deltaHigh5mCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[14])
    let! deltaLow5mCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[15])
    let! deltaClose5mCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[16])
    return {
        DeltaHigh1s = deltaHigh1sCol.Data :?> float[]
        DeltaLow1s = deltaLow1sCol.Data :?> float[]
        DeltaClose1s = deltaClose1sCol.Data :?> float[]
        DeltaHigh1m = deltaHigh1mCol.Data :?> float[]
        DeltaLow1m = deltaLow1mCol.Data :?> float[]
        DeltaClose1m = deltaClose1mCol.Data :?> float[]
        DeltaHigh5m = deltaHigh5mCol.Data :?> float[]
        DeltaLow5m = deltaLow5mCol.Data :?> float[]
        DeltaClose5m = deltaClose5mCol.Data :?> float[]
    }
}

let private addDeltasToDigest (td: MergingDigest) (deltas: DeltaArrays) =
    for i in 0 .. deltas.DeltaHigh1s.Length - 1 do
        td.Add(deltas.DeltaHigh1s.[i])
        td.Add(deltas.DeltaLow1s.[i])
        td.Add(deltas.DeltaClose1s.[i])
        td.Add(deltas.DeltaHigh1m.[i])
        td.Add(deltas.DeltaLow1m.[i])
        td.Add(deltas.DeltaClose1m.[i])
        td.Add(deltas.DeltaHigh5m.[i])
        td.Add(deltas.DeltaLow5m.[i])
        td.Add(deltas.DeltaClose5m.[i])

let private readRowGroupData (rowGroupReader: ParquetRowGroupReader) (schema: ParquetSchema) (rgIndex: int) = task {
    let! dayIdsCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[0])
    let! timesCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[1])
    let! opensCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[2])
    let! highsCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[3])
    let! lowsCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[4])
    let! closesCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[5])
    let! sessionsCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[6])
    let! trendsCol = rowGroupReader.ReadColumnAsync(schema.DataFields.[7])
    let! deltas = readDeltaArrays rowGroupReader schema
    return {
        Index = rgIndex
        DayIds = dayIdsCol.Data :?> int[]
        Times = timesCol.Data :?> int[]
        Opens = opensCol.Data :?> float[]
        Highs = highsCol.Data :?> float[]
        Lows = lowsCol.Data :?> float[]
        Closes = closesCol.Data :?> float[]
        Sessions = sessionsCol.Data :?> int[]
        Trends = trendsCol.Data :?> int[]
        Deltas = deltas
    }
}

let buildTDigestsFromParquet (inputPath: string) (compression: float) (numWorkers: int) = task {
    printfn "Building t-digests from %s..." inputPath
    
    use stream = File.OpenRead(inputPath)
    let! reader = ParquetReader.CreateAsync(stream)
    use reader = reader
    let rowGroupCount = reader.RowGroupCount
    let schema = reader.Schema
    
    printfn "  Using %d workers for %d row groups..." numWorkers rowGroupCount
    
    let channel = Channel.CreateBounded<DeltaArrays>(BoundedChannelOptions(numWorkers * 2))
    let mutable processed = 0
    
    let producer = task {
        for rgIndex in 0 .. rowGroupCount - 1 do
            use rowGroupReader = reader.OpenRowGroupReader(rgIndex)
            let! deltas = readDeltaArrays rowGroupReader schema
            do! channel.Writer.WriteAsync(deltas)
        channel.Writer.Complete()
    }
    
    let workers = [|
        for _ in 0 .. numWorkers - 1 ->
            task {
                let td = MergingDigest(compression)
                for deltas in channel.Reader.ReadAllAsync() do
                    addDeltasToDigest td deltas
                    let count = Interlocked.Increment(&processed)
                    if count % 100 = 0 then
                        printfn "  Processed %d / %d row groups" count rowGroupCount
                return td
            }
    |]
    
    do! producer
    let! digests = Task.WhenAll(workers)
    
    printfn "  Merging %d t-digests..." digests.Length
    let merged = MergingDigest(compression)
    merged.Add(digests |> Seq.cast<Digest>)
    
    printfn "Done. Processed %d row groups" rowGroupCount
    return { PriceDeltas = merged }
}

let saveTDigests (tds: TDigests) (outputPath: string) =
    use stream = File.Create(outputPath)
    use writer = new BinaryWriter(stream)
    tds.PriceDeltas.AsBytes(writer)
    printfn "Saved t-digests to %s" outputPath

let loadTDigests (inputPath: string) =
    use stream = File.OpenRead(inputPath)
    use reader = new BinaryReader(stream)
    { PriceDeltas = MergingDigest.FromBytes(reader) }

let applyCdf (td: MergingDigest) (values: float[]) =
    let result = Array.zeroCreate<float> values.Length
    for i in 0 .. values.Length - 1 do
        result.[i] <- td.Cdf(values.[i]) * 2.0 - 1.0
    result

type CdfLookupTable = {
    Min: float
    Max: float
    Step: float
    Values: float[]
}

let createCdfLookup (td: MergingDigest) (numBuckets: int) =
    let min = td.GetMin()
    let max = td.GetMax()
    let step = (max - min) / float (numBuckets - 1)
    let values = Array.init numBuckets (fun i ->
        let x = min + float i * step
        td.Cdf(x) * 2.0 - 1.0
    )
    { Min = min; Max = max; Step = step; Values = values }

let lookupValue (lookup: CdfLookupTable) (v: float) =
    let maxIdx = lookup.Values.Length - 1
    if v <= lookup.Min then
        lookup.Values.[0]
    elif v >= lookup.Max then
        lookup.Values.[maxIdx]
    else
        let idx = (v - lookup.Min) / lookup.Step
        let lo = int idx
        let hi = min (lo + 1) maxIdx
        let t = idx - float lo
        lookup.Values.[lo] * (1.0 - t) + lookup.Values.[hi] * t

let applyCdfWithLookup (lookup: CdfLookupTable) (values: float[]) = Array.map (lookupValue lookup) values

type private TransformedRowGroup = {
    Index: int
    DayIds: int[]
    Times: int[]
    Opens: float[]
    Highs: float[]
    Lows: float[]
    Closes: float[]
    Sessions: int[]
    Trends: int[]
    CdfHigh1s: float[]
    CdfLow1s: float[]
    CdfClose1s: float[]
    CdfHigh1m: float[]
    CdfLow1m: float[]
    CdfClose1m: float[]
    CdfHigh5m: float[]
    CdfLow5m: float[]
    CdfClose5m: float[]
}

let private applyTransform (lookup: CdfLookupTable) (data: RowGroupData) =
    { Index = data.Index
      DayIds = data.DayIds; Times = data.Times
      Opens = data.Opens; Highs = data.Highs; Lows = data.Lows; Closes = data.Closes
      Sessions = data.Sessions; Trends = data.Trends
      CdfHigh1s = applyCdfWithLookup lookup data.Deltas.DeltaHigh1s
      CdfLow1s = applyCdfWithLookup lookup data.Deltas.DeltaLow1s
      CdfClose1s = applyCdfWithLookup lookup data.Deltas.DeltaClose1s
      CdfHigh1m = applyCdfWithLookup lookup data.Deltas.DeltaHigh1m
      CdfLow1m = applyCdfWithLookup lookup data.Deltas.DeltaLow1m
      CdfClose1m = applyCdfWithLookup lookup data.Deltas.DeltaClose1m
      CdfHigh5m = applyCdfWithLookup lookup data.Deltas.DeltaHigh5m
      CdfLow5m = applyCdfWithLookup lookup data.Deltas.DeltaLow5m
      CdfClose5m = applyCdfWithLookup lookup data.Deltas.DeltaClose5m }

let private writeRowGroup (outSchema: ParquetSchema) (writer: ParquetWriter) (d: TransformedRowGroup) = task {
    use rowGroup = writer.CreateRowGroup()
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[0], d.DayIds))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[1], d.Times))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[2], d.Opens))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[3], d.Highs))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[4], d.Lows))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[5], d.Closes))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[6], d.Sessions))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[7], d.Trends))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[8], d.CdfHigh1s))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[9], d.CdfLow1s))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[10], d.CdfClose1s))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[11], d.CdfHigh1m))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[12], d.CdfLow1m))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[13], d.CdfClose1m))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[14], d.CdfHigh5m))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[15], d.CdfLow5m))
    do! rowGroup.WriteColumnAsync(DataColumn(outSchema.DataFields.[16], d.CdfClose5m))
}

let transformParquetWithCdf (inputPath: string) (tds: TDigests) (outputPath: string) (numWorkers: int) = task {
    printfn "Transforming %s with CDF..." inputPath
    
    let outSchema = ParquetSchema(
        DataField<int>("day_id"), DataField<int>("time"),
        DataField<float>("open"), DataField<float>("high"), DataField<float>("low"), DataField<float>("close"),
        DataField<int>("session"), DataField<int>("trend"),
        DataField<float>("cdf_high_1s"), DataField<float>("cdf_low_1s"), DataField<float>("cdf_close_1s"),
        DataField<float>("cdf_high_1m"), DataField<float>("cdf_low_1m"), DataField<float>("cdf_close_1m"),
        DataField<float>("cdf_high_5m"), DataField<float>("cdf_low_5m"), DataField<float>("cdf_close_5m")
    )
    
    use inStream = File.OpenRead(inputPath)
    let! reader = ParquetReader.CreateAsync(inStream)
    use reader = reader
    let rowGroupCount = reader.RowGroupCount
    let inSchema = reader.Schema
    
    printfn "  Using %d workers for %d row groups..." numWorkers rowGroupCount
    
    let lookupPriceDeltas = createCdfLookup tds.PriceDeltas (1 <<< 17) // 128k buckets
    
    let readChannel = Channel.CreateBounded<RowGroupData>(BoundedChannelOptions(numWorkers * 2))
    let writeChannel = Channel.CreateBounded<TransformedRowGroup>(BoundedChannelOptions(numWorkers * 2))
    let mutable processed = 0
    
    let producer = task {
        for rgIndex in 0 .. rowGroupCount - 1 do
            use rowGroupReader = reader.OpenRowGroupReader(rgIndex)
            let! data = readRowGroupData rowGroupReader inSchema rgIndex
            do! readChannel.Writer.WriteAsync(data)
        readChannel.Writer.Complete()
    }
    
    let workers = [|
        for _ in 0 .. numWorkers - 1 ->
            task {
                for data in readChannel.Reader.ReadAllAsync() do
                    let transformed = applyTransform lookupPriceDeltas data
                    do! writeChannel.Writer.WriteAsync(transformed)
                    let count = Interlocked.Increment(&processed)
                    if count % 100 = 0 then
                        printfn "  Transformed %d / %d row groups" count rowGroupCount
            }
    |]
    
    let workersDone = task {
        let! _ = Task.WhenAll(workers)
        writeChannel.Writer.Complete()
    }
    
    use outStream = File.Create(outputPath)
    let! writer = ParquetWriter.CreateAsync(outSchema, outStream)
    use writer = writer
    let pending = Collections.Generic.Dictionary<int, TransformedRowGroup>()
    let mutable nextToWrite = 0
    
    for data in writeChannel.Reader.ReadAllAsync() do
        pending.[data.Index] <- data
        while pending.ContainsKey(nextToWrite) do
            let d = pending.[nextToWrite]
            pending.Remove(nextToWrite) |> ignore
            do! writeRowGroup outSchema writer d
            nextToWrite <- nextToWrite + 1
    
    do! producer
    do! workersDone
    printfn "Done. Transformed %d row groups to %s" rowGroupCount outputPath
}
