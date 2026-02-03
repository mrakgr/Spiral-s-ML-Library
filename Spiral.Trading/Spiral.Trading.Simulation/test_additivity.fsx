#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

let medianPerSec = 50.0
let meanPerSec = 250.0

let sampleTradeCount (duration: float) =
    let medianCount = medianPerSec
    let meanCount = meanPerSec
    let mu = log(medianCount)
    let sigma = sqrt(2.0 * log(meanCount / medianCount))
    let mutable result = 0
    LogNormal(mu, sigma, rng).Sample() * duration

printfn "=== LogNormal Additivity Test ==="
printfn "median=%.0f, mean=%.0f per second" medianPerSec meanPerSec
printfn ""

let n = 10000

let single60 = Array.init n (fun _ -> sampleTradeCount 60.0)
let two30 = Array.init n (fun _ -> sampleTradeCount 30.0 + sampleTradeCount 30.0)

let mean1 = Array.average single60
let var1 = single60 |> Array.averageBy (fun x -> (x - mean1) ** 2.0)
let sorted1 = Array.sort single60
let median1 = sorted1.[n / 2]

let mean2 = Array.average two30
let var2 = two30 |> Array.averageBy (fun x -> (x - mean2) ** 2.0)
let sorted2 = Array.sort two30
let median2 = sorted2.[n / 2]

printfn "Single 60s:     median=%.0f, mean=%.0f, var=%.0f, stddev=%.0f" median1 mean1 var1 (sqrt var1)
printfn "Two 30s summed: median=%.0f, mean=%.0f, var=%.0f, stddev=%.0f" median2 mean2 var2 (sqrt var2)
printfn ""
printfn "Expected: median=%.0f, mean=%.0f" (medianPerSec * 60.0) (meanPerSec * 60.0)
printfn "Var ratio: %.2f (should be ~1.0 for additivity)" (var1 / var2)
