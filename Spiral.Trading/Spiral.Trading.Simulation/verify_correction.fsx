#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// Verify: E[correction * sqrt(activity)] = 1
// =============================================================================

let sigma = 1.0
let mu = -sigma * sigma / 2.0
let correction = exp(sigma * sigma / 8.0)

let samples = Array.init 100000 (fun _ -> LogNormal(mu, sigma, rng).Sample())

let avgSqrtActivity = samples |> Array.averageBy sqrt
let avgCorrectedSqrt = samples |> Array.averageBy (fun a -> correction * sqrt(a))

printfn "sigma = %.1f" sigma
printfn "correction = %.4f" correction
printfn "E[sqrt(activity)] = %.4f (theoretical: %.4f)" avgSqrtActivity (exp(-sigma*sigma/8.0))
printfn "E[correction * sqrt(activity)] = %.4f (should be 1.0)" avgCorrectedSqrt
