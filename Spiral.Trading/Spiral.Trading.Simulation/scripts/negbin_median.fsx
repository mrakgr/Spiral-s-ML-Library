#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

// =============================================================================
// Negative Binomial: Mean vs Median (varying mean and p)
//
// mean = r(1-p)/p  =>  r = mean * p / (1-p)
// variance = mean / p = mean + mean²/r
// =============================================================================

printfn "=== Negative Binomial: Mean vs Median (extreme p values) ==="
printfn ""
printfn "mean  | p      | r       | Variance   | StdDev | Median | Med/Mean"
printfn "------|--------|---------|------------|--------|--------|--------"

let mean = 100.0
for p in [0.9; 0.7; 0.5; 0.3; 0.1; 0.05; 0.01] do
    let r = mean * p / (1.0 - p)
    let variance = mean / p
    let stddev = sqrt variance
    let nb = NegativeBinomial(r, p)
    let mutable k = 0
    while nb.CumulativeDistribution(float k) < 0.5 do
        k <- k + 1
    let median = k
    printfn "%5.0f | %.2f   | %7.2f | %10.1f | %6.1f | %6d | %.3f" 
        mean p r variance stddev median (float median / mean)
