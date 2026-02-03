#r "nuget: Parquet.Net"
#r "nuget: T-Digest"
#r "nuget: MathNet.Numerics"
#r "nuget: FSharp.Control.TaskSeq"

#load "EpisodeMCMC.fs"
#load "PriceGeneration.fs"
#load "DatasetGeneration.fs"
#load "TDigestProcessing.fs"

open System
open System.IO
open Spiral.Trading.Simulation.DatasetGeneration
open Spiral.Trading.Simulation.TDigestProcessing
open Spiral.Trading.Simulation.EpisodeMCMC

let testDir = "../data/test_pipeline"
if not (Directory.Exists(testDir)) then
    Directory.CreateDirectory(testDir) |> ignore

let rawPath = Path.Combine(testDir, "raw.parquet")
let tdigestPath = rawPath + ".tdigests"
let cdfPath = Path.Combine(testDir, "cdf.parquet")

printfn "=== Testing T-Digest Pipeline ==="
printfn ""

// Step 1: Generate small dataset (10 days)
printfn "Step 1: Generating 10 days of raw data..."
let mcmcConfig = { MCMC.Iterations = 10000 }
let sessionConfig = SessionLevel.defaultConfig
let trendConfig = TrendLevel.defaultConfig
generateDataset 42 10 rawPath mcmcConfig sessionConfig trendConfig 100.0
printfn ""

// Step 2: Build t-digests
printfn "Step 2: Building t-digests from raw data (compression=%.0f)..." defaultCompression
let tds = (buildTDigestsFromParquet rawPath defaultCompression).Result
printfn ""

// Print some stats
printfn "T-digest stats:"
printfn "  PriceDeltas: count=%d, q01=%.4f, q50=%.4f, q99=%.4f" 
    (tds.PriceDeltas.Size()) 
    (tds.PriceDeltas.Quantile(0.01)) 
    (tds.PriceDeltas.Quantile(0.50))
    (tds.PriceDeltas.Quantile(0.99))
printfn ""

// Step 3: Save t-digests
printfn "Step 3: Saving t-digests..."
saveTDigests tds tdigestPath
printfn ""

// Step 4: Load t-digests back
printfn "Step 4: Loading t-digests..."
let tdsLoaded = loadTDigests tdigestPath
printfn "  Loaded successfully, PriceDeltas count=%d" (tdsLoaded.PriceDeltas.Size())
printfn ""

// Step 5: Transform with CDF
printfn "Step 5: Transforming with CDF..."
(transformParquetWithCdf rawPath tdsLoaded cdfPath).Wait()
printfn ""

// Step 6: Verify output
printfn "Step 6: Verifying output..."
let rawSize = FileInfo(rawPath).Length
let cdfSize = FileInfo(cdfPath).Length
let tdigestSize = FileInfo(tdigestPath).Length
printfn "  Raw parquet: %d bytes" rawSize
printfn "  CDF parquet: %d bytes" cdfSize
printfn "  T-digests: %d bytes" tdigestSize
printfn ""

// Read a sample from CDF output to verify values are in [-1, 1]
printfn "Step 7: Checking CDF values are in [-1, 1]..."
open Parquet
use stream = File.OpenRead(cdfPath)
use reader = ParquetReader.CreateAsync(stream).Result
use rgReader = reader.OpenRowGroupReader(0)
let cdfHigh1s = (rgReader.ReadColumnAsync(reader.Schema.DataFields.[8]).Result).Data :?> float[]
let cdfLow1s = (rgReader.ReadColumnAsync(reader.Schema.DataFields.[9]).Result).Data :?> float[]
let cdfClose1s = (rgReader.ReadColumnAsync(reader.Schema.DataFields.[10]).Result).Data :?> float[]

let minVal = Array.min [| Array.min cdfHigh1s; Array.min cdfLow1s; Array.min cdfClose1s |]
let maxVal = Array.max [| Array.max cdfHigh1s; Array.max cdfLow1s; Array.max cdfClose1s |]
printfn "  CDF value range: [%.4f, %.4f]" minVal maxVal

if minVal >= -1.0 && maxVal <= 1.0 then
    printfn "  SUCCESS: All values in [-1, 1]"
else
    printfn "  WARNING: Values outside [-1, 1]"

printfn ""
printfn "=== Pipeline test complete ==="
