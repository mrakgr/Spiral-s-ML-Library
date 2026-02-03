#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

// Check MathNet's NegativeBinomial parameterization
let r = 100.0
let p = 0.5

let nb = NegativeBinomial(r, p)
printfn "MathNet NegativeBinomial(r=%.0f, p=%.1f):" r p
printfn "  .Mean = %.1f" nb.Mean
printfn "  .Variance = %.1f" nb.Variance
printfn ""

// Two common parameterizations:
// 1. "failures until r successes": mean = r(1-p)/p
// 2. "successes until r failures": mean = rp/(1-p)  [this seems to be MathNet's]

printfn "Expected mean if r(1-p)/p: %.1f" (r * (1.0 - p) / p)
printfn "Expected mean if rp/(1-p): %.1f" (r * p / (1.0 - p))
