#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// Our Gamma-Poisson mixture implementation
let sampleGammaPoisson (rng: Random) (r: float) (p: float) =
    // NegBin(r,p) = Poisson(lambda) where lambda ~ Gamma(r, (1-p)/p)
    let lambda = Gamma(r, (1.0 - p) / p, rng).Sample()
    Poisson(lambda, rng).Sample()

let r = 100.0
let p = 0.5

printfn "Target: mean = r(1-p)/p = %.1f" (r * (1.0 - p) / p)
printfn ""

let samples = Array.init 10000 (fun _ -> sampleGammaPoisson rng r p)
printfn "Gamma-Poisson mixture (10000 samples):"
printfn "  Mean: %.1f" (Array.averageBy float samples)
printfn "  Variance: %.1f" (let m = Array.averageBy float samples in samples |> Array.averageBy (fun x -> (float x - m) ** 2.0))
