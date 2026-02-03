#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

let sampleTradeCount (rate: float) (dispersionExp: float) (duration: float) =
    let mean = rate * duration
    if dispersionExp < 0.01 then
        Poisson(mean, rng).Sample()
    else
        let p = Math.Pow(2.0, -dispersionExp)
        let r = mean * p / (1.0 - p)
        let gammaRate = p / (1.0 - p)
        let lambda = Gamma(r, gammaRate, rng).Sample()
        Poisson(lambda, rng).Sample()

let rate = 50.0
let duration = 60.0

printfn "Expected mean: %.0f" (rate * duration)
printfn ""

for dispExp in [0.0; 1.0; 2.0; 3.0] do
    let samples = Array.init 1000 (fun _ -> sampleTradeCount rate dispExp duration)
    let mean = Array.averageBy float samples
    let std = sqrt (samples |> Array.averageBy (fun x -> (float x - mean) ** 2.0))
    printfn "dispExp=%.1f: mean=%.0f, std=%.0f" dispExp mean std
