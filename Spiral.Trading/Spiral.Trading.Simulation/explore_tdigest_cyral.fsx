#r "nuget: T-Digest"
#r "nuget: MathNet.Numerics"

open System
open TDigest
open MathNet.Numerics.Distributions

let rng = Random(42)
let td = MergingDigest(100.0)
let normal = Normal(0.0, 1.0)

// Add normal(0,1) data
for _ in 1..100000 do
    td.Add(normal.Sample())

printfn "Count: %d" (int64 (td.Size()))

printfn "\nCDF comparison (t-digest vs true normal CDF):"
for x in [-2.0; -1.0; 0.0; 1.0; 2.0] do
    let td_cdf = td.Cdf(x)
    let true_cdf = normal.CumulativeDistribution(x)
    printfn "  x=%2.0f: t-digest=%.4f, true=%.4f, diff=%.4f" x td_cdf true_cdf (td_cdf - true_cdf)

printfn "\nQuantile comparison (t-digest vs true normal quantile):"
for p in [0.01; 0.1; 0.5; 0.9; 0.99] do
    let td_q = td.Quantile(p)
    let true_q = normal.InverseCumulativeDistribution(p)
    printfn "  p=%.2f: t-digest=%.4f, true=%.4f, diff=%.4f" p td_q true_q (td_q - true_q)

printfn "\nThe list of centroids:"
for c in td.Centroids() do
    printfn "  %A" c