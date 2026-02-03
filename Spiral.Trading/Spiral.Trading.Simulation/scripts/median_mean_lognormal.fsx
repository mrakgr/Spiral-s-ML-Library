#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// LogNormal parameterized by median and mean
//
// median = exp(mu)  =>  mu = ln(median)
// mean = exp(mu + sigma²/2)  =>  sigma = sqrt(2 * ln(mean/median))
// =============================================================================

let lognormalFromMedianMean (median: float) (mean: float) =
    if mean < median then failwith "mean must be >= median for LogNormal"
    let mu = log(median)
    let sigma = sqrt(2.0 * log(mean / median))
    (mu, sigma)

printfn "=== LogNormal: Median/Mean Parameterization ==="
printfn ""

// Show current problem: fixed mean=100, varying sigma
printfn "Current approach (BaseSize=100, varying sigma):"
printfn "sigma | mu      | median | mean"
printfn "------|---------|--------|-----"
for sigma in [0.6; 0.8; 1.0; 1.2] do
    let mu = -sigma * sigma / 2.0  // E[X] = 1, so actual mean = BaseSize * 1 = 100
    let median = exp(mu) * 100.0   // median = exp(mu) * BaseSize
    printfn " %.1f  | %6.3f  | %5.1f  | 100.0" sigma mu median

printfn ""
printfn "Problem: median decreases as sigma increases!"
printfn ""

// New approach: fixed median, varying mean
printfn "=== New approach: fixed median=100, varying mean ==="
printfn ""
printfn "median | mean | mu     | sigma | mean/median"
printfn "-------|------|--------|-------|------------"
for mean in [100.0; 120.0; 150.0; 200.0; 300.0] do
    let median = 100.0
    let mu, sigma = lognormalFromMedianMean median mean
    printfn "  %3.0f  | %3.0f  | %5.2f  | %5.3f | %.1fx" median mean mu sigma (mean/median)

printfn ""
printfn "=== Empirical verification ==="
printfn ""

// Test with median=100, mean=200
let median, mean = 100.0, 200.0
let mu, sigma = lognormalFromMedianMean median mean
let samples = Array.init 100000 (fun _ -> LogNormal(mu, sigma, rng).Sample())
let empiricalMedian = (Array.sort samples).[50000]
let empiricalMean = Array.average samples

printfn "Target: median=%.0f, mean=%.0f" median mean
printfn "Empirical: median=%.1f, mean=%.1f" empiricalMedian empiricalMean
