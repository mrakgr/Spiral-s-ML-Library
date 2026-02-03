#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// Correct: Gamma(shape=r, rate=p/(1-p))
let sampleGammaPoisson (rng: Random) (r: float) (p: float) =
    let rate = p / (1.0 - p)
    let lambda = Gamma(r, rate, rng).Sample()
    Poisson(lambda, rng).Sample()

let mean = 100.0

for p in [0.5; 0.25; 0.1; 0.05] do
    let r = mean * p / (1.0 - p)
    let variance = mean / p
    let samples = Array.init 50 (fun _ -> sampleGammaPoisson rng r p)
    
    printfn "=== p=%.2f, r=%.2f, variance=%.0f, stddev=%.1f ===" p r variance (sqrt variance)
    for i in 0 .. 4 do
        let row = samples.[i*10 .. i*10+9] |> Array.map (sprintf "%4d") |> String.concat " "
        printfn "%s" row
    printfn "Min=%d, Max=%d" (Array.min samples) (Array.max samples)
    printfn ""
