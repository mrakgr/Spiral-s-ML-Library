#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// Gamma(shape=r, scale=(1-p)/p) has mean = r*(1-p)/p
// MathNet: Gamma(shape, rate) where rate = 1/scale = p/(1-p)
let sampleGammaPoissonFixed (rng: Random) (r: float) (p: float) =
    let rate = p / (1.0 - p)
    let lambda = Gamma(r, rate, rng).Sample()
    Poisson(lambda, rng).Sample()

let mean = 100.0
let p = 0.5
let r = mean * p / (1.0 - p)

printfn "target mean=%.0f, p=%.1f, r=%.1f" mean p r
printfn "Gamma rate = p/(1-p) = %.2f" (p / (1.0 - p))
printfn "Gamma mean = r/rate = %.1f" (r / (p / (1.0 - p)))

let samples = Array.init 10000 (fun _ -> sampleGammaPoissonFixed rng r p)
printfn "Empirical mean: %.1f" (Array.averageBy float samples)
