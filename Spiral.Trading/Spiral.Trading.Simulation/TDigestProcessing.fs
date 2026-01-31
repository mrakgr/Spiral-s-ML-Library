module Spiral.Trading.Simulation.TDigestProcessing

open System
open System.IO
open Parquet
open Parquet.Schema
open Parquet.Data
open TDigest

open System.Threading.Tasks

type TDigests = {
    PriceDeltas: MergingDigest
}

let createTDigests (compression: float) =
    { PriceDeltas = MergingDigest(compression) }

let defaultCompression = 4096.0 // 2^12

let buildTDigestsFromParquet (inputPath: string) (compression: float) = task {
    printfn "Building t-digests from %s..." inputPath
    
    let tds = createTDigests compression
    use stream = File.OpenRead(inputPath)
    let! reader = ParquetReader.CreateAsync(stream)
    use reader = reader
    
    let mutable rowGroupsProcessed = 0
    for rgIndex in 0 .. reader.RowGroupCount - 1 do
        use rowGroupReader = reader.OpenRowGroupReader(rgIndex)
        
        let! deltaHigh1sCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[8])
        let! deltaLow1sCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[9])
        let! deltaClose1sCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[10])
        let! deltaHigh1mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[11])
        let! deltaLow1mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[12])
        let! deltaClose1mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[13])
        let! deltaHigh5mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[14])
        let! deltaLow5mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[15])
        let! deltaClose5mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[16])
        
        let deltaHigh1s = deltaHigh1sCol.Data :?> float[]
        let deltaLow1s = deltaLow1sCol.Data :?> float[]
        let deltaClose1s = deltaClose1sCol.Data :?> float[]
        let deltaHigh1m = deltaHigh1mCol.Data :?> float[]
        let deltaLow1m = deltaLow1mCol.Data :?> float[]
        let deltaClose1m = deltaClose1mCol.Data :?> float[]
        let deltaHigh5m = deltaHigh5mCol.Data :?> float[]
        let deltaLow5m = deltaLow5mCol.Data :?> float[]
        let deltaClose5m = deltaClose5mCol.Data :?> float[]
        
        for i in 0 .. deltaHigh1s.Length - 1 do
            tds.PriceDeltas.Add(deltaHigh1s.[i])
            tds.PriceDeltas.Add(deltaLow1s.[i])
            tds.PriceDeltas.Add(deltaClose1s.[i])
            tds.PriceDeltas.Add(deltaHigh1m.[i])
            tds.PriceDeltas.Add(deltaLow1m.[i])
            tds.PriceDeltas.Add(deltaClose1m.[i])
            tds.PriceDeltas.Add(deltaHigh5m.[i])
            tds.PriceDeltas.Add(deltaLow5m.[i])
            tds.PriceDeltas.Add(deltaClose5m.[i])
        
        rowGroupsProcessed <- rowGroupsProcessed + 1
        if rowGroupsProcessed % 500 = 0 then
            printfn "  Processed %d row groups" rowGroupsProcessed
    
    printfn "Done. Processed %d row groups" rowGroupsProcessed
    return tds
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

let transformParquetWithCdf (inputPath: string) (tds: TDigests) (outputPath: string) = task {
    printfn "Transforming %s with CDF..." inputPath
    
    let schema = ParquetSchema(
        DataField<int>("day_id"),
        DataField<int>("time"),
        DataField<float>("open"),
        DataField<float>("high"),
        DataField<float>("low"),
        DataField<float>("close"),
        DataField<int>("session"),
        DataField<int>("trend"),
        DataField<float>("cdf_high_1s"),
        DataField<float>("cdf_low_1s"),
        DataField<float>("cdf_close_1s"),
        DataField<float>("cdf_high_1m"),
        DataField<float>("cdf_low_1m"),
        DataField<float>("cdf_close_1m"),
        DataField<float>("cdf_high_5m"),
        DataField<float>("cdf_low_5m"),
        DataField<float>("cdf_close_5m")
    )
    
    use inStream = File.OpenRead(inputPath)
    let! reader = ParquetReader.CreateAsync(inStream)
    use reader = reader
    use outStream = File.Create(outputPath)
    let! writer = ParquetWriter.CreateAsync(schema, outStream)
    use writer = writer
    
    let mutable rowGroupsProcessed = 0
    for rgIndex in 0 .. reader.RowGroupCount - 1 do
        use rowGroupReader = reader.OpenRowGroupReader(rgIndex)
        
        let! dayIdsCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[0])
        let! timesCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[1])
        let! opensCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[2])
        let! highsCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[3])
        let! lowsCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[4])
        let! closesCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[5])
        let! sessionsCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[6])
        let! trendsCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[7])
        let! deltaHigh1sCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[8])
        let! deltaLow1sCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[9])
        let! deltaClose1sCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[10])
        let! deltaHigh1mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[11])
        let! deltaLow1mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[12])
        let! deltaClose1mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[13])
        let! deltaHigh5mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[14])
        let! deltaLow5mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[15])
        let! deltaClose5mCol = rowGroupReader.ReadColumnAsync(reader.Schema.DataFields.[16])
        
        let dayIds = dayIdsCol.Data :?> int[]
        let times = timesCol.Data :?> int[]
        let opens = opensCol.Data :?> float[]
        let highs = highsCol.Data :?> float[]
        let lows = lowsCol.Data :?> float[]
        let closes = closesCol.Data :?> float[]
        let sessions = sessionsCol.Data :?> int[]
        let trends = trendsCol.Data :?> int[]
        let deltaHigh1s = deltaHigh1sCol.Data :?> float[]
        let deltaLow1s = deltaLow1sCol.Data :?> float[]
        let deltaClose1s = deltaClose1sCol.Data :?> float[]
        let deltaHigh1m = deltaHigh1mCol.Data :?> float[]
        let deltaLow1m = deltaLow1mCol.Data :?> float[]
        let deltaClose1m = deltaClose1mCol.Data :?> float[]
        let deltaHigh5m = deltaHigh5mCol.Data :?> float[]
        let deltaLow5m = deltaLow5mCol.Data :?> float[]
        let deltaClose5m = deltaClose5mCol.Data :?> float[]
        
        use rowGroup = writer.CreateRowGroup()
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[0], dayIds))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[1], times))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[2], opens))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[3], highs))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[4], lows))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[5], closes))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[6], sessions))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[7], trends))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[8], applyCdf tds.PriceDeltas deltaHigh1s))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[9], applyCdf tds.PriceDeltas deltaLow1s))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[10], applyCdf tds.PriceDeltas deltaClose1s))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[11], applyCdf tds.PriceDeltas deltaHigh1m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[12], applyCdf tds.PriceDeltas deltaLow1m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[13], applyCdf tds.PriceDeltas deltaClose1m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[14], applyCdf tds.PriceDeltas deltaHigh5m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[15], applyCdf tds.PriceDeltas deltaLow5m))
        do! rowGroup.WriteColumnAsync(DataColumn(schema.DataFields.[16], applyCdf tds.PriceDeltas deltaClose5m))
        
        rowGroupsProcessed <- rowGroupsProcessed + 1
        if rowGroupsProcessed % 500 = 0 then
            printfn "  Transformed %d row groups" rowGroupsProcessed
    
    printfn "Done. Transformed %d row groups to %s" rowGroupsProcessed outputPath
}
