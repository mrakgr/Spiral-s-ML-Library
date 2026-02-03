#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

let medianPerSec = 50.0
let meanPerSec = 60.0

let sampleTradeCount (duration: float) =
    let medianCount = medianPerSec * duration
    let meanCount = meanPerSec * duration
    let mu = log(medianCount)
    let sigma = sqrt(2.0 * log(meanCount / medianCount))
    LogNormal(mu, sigma, rng).Sample()

printfn "=== Additivity Test ==="
printfn ""

let n = 10000

let single60 = Array.init n (fun _ -> sampleTradeCount 60.0)
let two30 = Array.init n (fun _ -> sampleTradeCount 30.0 + sampleTradeCount 30.0)

let mean1 = Array.average single60
let var1 = single60 |> Array.averageBy (fun x -> (x - mean1) ** 2.0)

let mean2 = Array.average two30
let var2 = two30 |> Array.averageBy (fun x -> (x - mean2) ** 2.0)

printfn "Single 60s:     mean=%.0f, var=%.0f, stddev=%.0f" mean1 var1 (sqrt var1)
printfn "Two 30s summed: mean=%.0f, var=%.0f, stddev=%.0f" mean2 var2 (sqrt var2)
printfn ""
printfn "For additivity, variances should be equal."
printfn "Var ratio: %.2f (should be ~1.0)" (var1 / var2)
