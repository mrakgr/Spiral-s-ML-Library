#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

// =============================================================================
// Hypothesis: We're not simulating the full duration
// The last timestamp is < durationSeconds on average
// =============================================================================

let rng = Random(42)
let durationSeconds = 60.0
let tradesPerRun = 100

let generateTimestamps (rng: Random) duration count =
    let ts = Array.init count (fun _ -> rng.NextDouble() * duration)
    Array.sortInPlace ts
    ts

// Check average last timestamp
let runs = 100000
let mutable sumLastTs = 0.0
for _ in 1 .. runs do
    let ts = generateTimestamps rng durationSeconds tradesPerRun
    sumLastTs <- sumLastTs + ts.[ts.Length - 1]

let avgLastTs = sumLastTs / float runs
printfn "Average last timestamp: %.4f seconds" avgLastTs
printfn "Expected duration: %.4f seconds" durationSeconds
printfn "Ratio: %.6f" (avgLastTs / durationSeconds)
printfn ""
printfn "Expected ratio for order statistic: n/(n+1) = %d/%d = %.6f" 
    tradesPerRun (tradesPerRun + 1) (float tradesPerRun / float (tradesPerRun + 1))
