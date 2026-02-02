#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// Comparing distribution shapes for trade sizes
// 
// Requirements:
// - Positive support (sizes > 0)
// - A "hump" (mode > 0, not at the boundary)
// - Heavy right tail (occasional large trades)
// =============================================================================

// Sample and bin a distribution
let histogram (samples: float[]) (binEdges: float[]) =
    let counts = Array.zeroCreate (binEdges.Length - 1)
    for s in samples do
        for i in 0 .. binEdges.Length - 2 do
            if s >= binEdges.[i] && s < binEdges.[i+1] then
                counts.[i] <- counts.[i] + 1
    counts

let printHistogram (name: string) (binEdges: float[]) (counts: int[]) =
    let maxCount = Array.max counts
    let scale = 40.0 / float maxCount
    printfn "%s:" name
    for i in 0 .. counts.Length - 1 do
        let lo, hi = binEdges.[i], binEdges.[i+1]
        let bar = String.replicate (int (float counts.[i] * scale)) "#"
        printfn "  [%5.1f-%5.1f) %5d %s" lo hi counts.[i] bar
    printfn ""

let n = 10000
let bins = [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 8.0; 10.0; 15.0; 20.0; 30.0; 50.0; infinity |]

// 1. Pareto - mode at minimum, no hump
printfn "=== Distribution Shape Comparison ==="
printfn ""
let pareto = Array.init n (fun _ -> Pareto(1.0, 1.5, rng).Sample())
printHistogram "Pareto(min=1, alpha=1.5) - mode at boundary" bins (histogram pareto bins)

// 2. LogNormal - has a hump, heavy tail
let lognormal = Array.init n (fun _ -> LogNormal(1.0, 0.8, rng).Sample())
printHistogram "LogNormal(mu=1, sigma=0.8) - has hump" bins (histogram lognormal bins)

// 3. Gamma - has a hump, lighter tail than lognormal
let gamma = Array.init n (fun _ -> Gamma(2.0, 2.0, rng).Sample())
printHistogram "Gamma(shape=2, rate=2) - has hump" bins (histogram gamma bins)

// 4. Shifted/Scaled LogNormal for typical trade sizes
// If we want mode around 100 shares with heavy tail
let mu, sigma = 4.5, 1.0  // mode = exp(mu - sigma^2) ~ 33, mean = exp(mu + sigma^2/2) ~ 148
let lognormalTrades = Array.init n (fun _ -> LogNormal(mu, sigma, rng).Sample())
let tradeBins = [| 0.0; 10.0; 25.0; 50.0; 100.0; 200.0; 500.0; 1000.0; 2000.0; 5000.0; infinity |]
printHistogram "LogNormal(mu=4.5, sigma=1.0) - realistic trade sizes" tradeBins (histogram lognormalTrades tradeBins)

// Summary stats
printfn "=== Summary Statistics ==="
printfn ""
printfn "Distribution          | Mean    | Median  | Mode    | Max"
printfn "----------------------|---------|---------|---------|--------"
let stats name (samples: float[]) modeApprox =
    let sorted = Array.sort samples
    let mean = Array.average samples
    let median = sorted.[n/2]
    let max = Array.max samples
    printfn "%-21s | %7.1f | %7.1f | %7.1f | %7.0f" name mean median modeApprox max

stats "Pareto" pareto 1.0
stats "LogNormal(0.8)" lognormal (exp(1.0 - 0.8*0.8))
stats "Gamma" gamma 0.5  // mode = (shape-1)/rate for shape>1
stats "LogNormal(trades)" lognormalTrades (exp(mu - sigma*sigma))
