#r "nuget: T-Digest.NET"
#r "nuget: MathNet.Numerics"

open System
open TDigestNet
open MathNet.Numerics.Distributions

let rng = Random(42)
let td = TDigest()
let normal = Normal(0.0, 1.0)

// Add normal(0,1) data
for _ in 1..100000 do
    td.Add(normal.Sample())

printfn "Count: %d" (int td.Count)

// CDF via binary search on Quantile
let cdf (td: TDigest) (x: float) =
    if x <= td.Min then 0.0
    elif x >= td.Max then 1.0
    else
        let mutable lo, hi = 0.0, 1.0
        for _ in 1..50 do
            let mid = (lo + hi) / 2.0
            if td.Quantile(mid) <= x then lo <- mid else hi <- mid
        (lo + hi) / 2.0

printfn "\nCDF comparison (t-digest vs true normal CDF):"
for x in [-2.0; -1.0; 0.0; 1.0; 2.0] do
    let td_cdf = cdf td x
    let td_quantile = td.Quantile x
    let true_cdf = normal.CumulativeDistribution(x)
    printfn "  x=%2.0f: t-digest=%.4f, true=%.4f, diff=%.4f" x td_cdf true_cdf (td_cdf - true_cdf)
    printfn "  quantile=%.4f" td_quantile
