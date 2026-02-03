#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// Gamma-Poisson mixture (correct NegBin implementation)
let sampleNegBin (mean: float) (p: float) =
    let r = mean * p / (1.0 - p)
    let gammaRate = p / (1.0 - p)
    let lambda = Gamma(r, gammaRate, rng).Sample()
    Poisson(lambda, rng).Sample()

let meanPerSec = 60.0
let p = 0.5  // dispersion parameter

printfn "=== NegBin Additivity Test ==="
printfn "mean=%.0f per second, p=%.1f" meanPerSec p
printfn ""

let n = 10000

let single60 = Array.init n (fun _ -> sampleNegBin (meanPerSec * 60.0) p)
let two30 = Array.init n (fun _ -> sampleNegBin (meanPerSec * 30.0) p + sampleNegBin (meanPerSec * 30.0) p)

let mean1 = Array.averageBy float single60
let var1 = single60 |> Array.averageBy (fun x -> (float x - mean1) ** 2.0)

let mean2 = Array.averageBy float two30
let var2 = two30 |> Array.averageBy (fun x -> (float x - mean2) ** 2.0)

printfn "Single 60s:     mean=%.0f, var=%.0f, stddev=%.0f" mean1 var1 (sqrt var1)
printfn "Two 30s summed: mean=%.0f, var=%.0f, stddev=%.0f" mean2 var2 (sqrt var2)
printfn ""
printfn "Var ratio: %.2f (should be ~1.0 for additivity)" (var1 / var2)
