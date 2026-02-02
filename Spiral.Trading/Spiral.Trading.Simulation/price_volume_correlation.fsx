#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// Problem: Sample (price_change, size) pairs where:
// - size ~ Pareto (heavy tail)
// - price_change ~ Normal (or similar)
// - E[|price_change|] ~ sqrt(size)  (the square-root impact law)
// =============================================================================

// Option 1: 2D Gaussian with correlation
// Problem: Gaussian marginals, but size must be positive with heavy tail

// Option 2: Copula approach
// - Sample marginals independently from desired distributions
// - Use a copula to induce correlation structure
// - Gaussian copula is common, but there are others (Clayton, Gumbel, Frank)

// Option 3: Conditional sampling
// - First sample size from Pareto
// - Then sample price_change conditioned on size: price_change ~ N(0, sigma * sqrt(size))
// This directly encodes the sqrt relationship in the variance!

// Option 4: Stochastic volatility
// - Sample a latent "activity" variable
// - Both size and |price_change| depend on this activity

// =============================================================================
// Let's explore Option 3: Conditional sampling
// =============================================================================

printfn "=== Option 3: Conditional Sampling ==="
printfn "price_change ~ N(0, baseVol * sqrt(size))"
printfn ""

let baseVol = 0.001  // Base volatility
let paretoAlpha = 1.5
let paretoMin = 1.0

let sampleConditional () =
    let size = Pareto(paretoMin, paretoAlpha, rng).Sample()
    let vol = baseVol * sqrt(size)
    let priceChange = Normal(0.0, vol, rng).Sample()
    (size, priceChange)

// Generate samples
let samples = Array.init 10000 (fun _ -> sampleConditional())

// Check the relationship: bin by size, compute E[|price_change|] per bin
let bins = [| 1.0; 2.0; 4.0; 8.0; 16.0; 32.0; 64.0; 128.0; infinity |]

printfn "Size Range        | Count | E[|dP|]   | sqrt(midpoint) | Ratio"
printfn "------------------|-------|-----------|----------------|------"

for i in 0 .. bins.Length - 2 do
    let lo, hi = bins.[i], bins.[i+1]
    let inBin = samples |> Array.filter (fun (s, _) -> s >= lo && s < hi)
    if inBin.Length > 0 then
        let avgAbsChange = inBin |> Array.averageBy (fun (_, dp) -> abs dp)
        let midpoint = if hi = infinity then lo * 2.0 else (lo + hi) / 2.0
        let expectedRatio = sqrt(midpoint)
        let ratio = avgAbsChange / (baseVol * sqrt(midpoint))
        printfn "[%5.0f, %5.0f)    | %5d | %.6f | %.3f          | %.3f" 
            lo hi inBin.Length avgAbsChange (sqrt midpoint) ratio
