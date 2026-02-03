#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// Correct Gamma-Poisson: lambda ~ Gamma with mean = (1-p)/p
// Gamma(shape, rate) has mean = shape/rate
// So rate = shape / mean = r / ((1-p)/p) = rp/(1-p)
let sampleGammaPoissonFixed (rng: Random) (r: float) (p: float) =
    let gammaMean = (1.0 - p) / p
    let rate = r / gammaMean  // = rp/(1-p)
    let lambda = Gamma(r, rate, rng).Sample()
    Poisson(lambda, rng).Sample()

let mean = 100.0
let p = 0.5
let r = mean * p / (1.0 - p)

printfn "Fixed Gamma-Poisson: target mean=%.0f, p=%.1f, r=%.1f" mean p r
let samples = Array.init 10000 (fun _ -> sampleGammaPoissonFixed rng r p)
printfn "Empirical mean: %.1f" (Array.averageBy float samples)
