#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// Test: Is LogNormal additive in mean and variance?
// =============================================================================

let mu = 4.0
let sigma = 1.0

printfn "LogNormal(mu=%.1f, sigma=%.1f)" mu sigma
printfn "Theoretical: mean=%.1f, var=%.1f" 
    (exp(mu + sigma*sigma/2.0)) 
    ((exp(sigma*sigma) - 1.0) * exp(2.0*mu + sigma*sigma))
printfn ""

// Single sample
let single = Array.init 10000 (fun _ -> LogNormal(mu, sigma, rng).Sample())
let singleMean = Array.average single
let singleVar = single |> Array.averageBy (fun x -> (x - singleMean) ** 2.0)

printfn "Single sample (n=10000):"
printfn "  Mean: %.1f, Var: %.1f" singleMean singleVar
printfn ""

// Sum of 2 samples
let sum2 = Array.init 10000 (fun _ -> 
    LogNormal(mu, sigma, rng).Sample() + LogNormal(mu, sigma, rng).Sample())
let sum2Mean = Array.average sum2
let sum2Var = sum2 |> Array.averageBy (fun x -> (x - sum2Mean) ** 2.0)

printfn "Sum of 2 samples:"
printfn "  Mean: %.1f (expected: %.1f)" sum2Mean (2.0 * singleMean)
printfn "  Var: %.1f (expected: %.1f)" sum2Var (2.0 * singleVar)
printfn ""

// Sum of 50 samples
let sum5 = Array.init 10000 (fun _ -> 
    Array.init 50 (fun _ -> LogNormal(mu, sigma, rng).Sample()) |> Array.sum)
let sum50Mean = Array.average sum5
let sum50Var = sum5 |> Array.averageBy (fun x -> (x - sum50Mean) ** 2.0)

printfn "Sum of 50 samples:"
printfn "  Mean: %.1f (expected: %.1f)" sum50Mean (50.0 * singleMean)
printfn "  Var: %.1f (expected: %.1f)" sum50Var (50.0 * singleVar)
