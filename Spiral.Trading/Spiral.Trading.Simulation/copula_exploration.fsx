#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// Copula: A way to couple marginal distributions with a dependence structure
//
// Steps:
// 1. Sample correlated uniforms (u, v) from the copula
// 2. Transform via inverse CDF: x = F_X^(-1)(u), y = F_Y^(-1)(v)
// =============================================================================

// Gaussian Copula: use 2D normal to generate correlated uniforms
let sampleGaussianCopula (rho: float) : float * float =
    // Sample from 2D standard normal with correlation rho
    let z1 = Normal.Sample(rng, 0.0, 1.0)
    let z2 = rho * z1 + sqrt(1.0 - rho * rho) * Normal.Sample(rng, 0.0, 1.0)
    // Transform to uniforms via normal CDF
    let u = Normal.CDF(0.0, 1.0, z1)
    let v = Normal.CDF(0.0, 1.0, z2)
    (u, v)

// Inverse CDFs for our marginals
let paretoInverseCDF (minVal: float) (alpha: float) (u: float) : float =
    minVal / Math.Pow(1.0 - u, 1.0 / alpha)

let normalInverseCDF (mean: float) (stdDev: float) (u: float) : float =
    Normal.InvCDF(mean, stdDev, u)

// =============================================================================
// Example: Sample (size ~ Pareto, priceChange ~ Normal) with Gaussian copula
// =============================================================================

let paretoMin = 1.0
let paretoAlpha = 1.5
let priceMean = 0.0
let priceStdDev = 0.001

printfn "=== Gaussian Copula Example ==="
printfn "Marginals: size ~ Pareto(min=1, alpha=1.5), priceChange ~ N(0, 0.001)"
printfn ""

for rho in [0.0; 0.3; 0.6; 0.9] do
    let samples = Array.init 5000 (fun _ ->
        let u, v = sampleGaussianCopula rho
        let size = paretoInverseCDF paretoMin paretoAlpha u
        let priceChange = normalInverseCDF priceMean priceStdDev v
        (size, priceChange))
    
    // Compute correlation between size and |priceChange|
    let sizes = samples |> Array.map fst
    let absChanges = samples |> Array.map (snd >> abs)
    
    let meanSize = Array.average sizes
    let meanAbs = Array.average absChanges
    let cov = Array.map2 (fun s a -> (s - meanSize) * (a - meanAbs)) sizes absChanges |> Array.average
    let stdSize = sqrt (Array.averageBy (fun s -> (s - meanSize) ** 2.0) sizes)
    let stdAbs = sqrt (Array.averageBy (fun a -> (a - meanAbs) ** 2.0) absChanges)
    let corr = cov / (stdSize * stdAbs)
    
    printfn "rho = %.1f -> Corr(size, |priceChange|) = %.3f" rho corr

printfn ""
printfn "Note: Gaussian copula gives linear correlation in the transformed space,"
printfn "but doesn't directly encode sqrt(size) relationship."
