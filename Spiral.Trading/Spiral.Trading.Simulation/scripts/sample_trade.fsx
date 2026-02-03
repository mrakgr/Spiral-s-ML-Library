#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

/// Samples trade count from Negative Binomial distribution.
/// rate: trades per second
/// dispersionExp: 0 = Poisson-like, 1 = 2x variance, 2 = 4x variance, etc.
/// duration: episode length in seconds
let sampleTradeCount (rng: Random) (rate: float) (dispersionExp: float) (duration: float) =
    let p = Math.Pow(2.0, -dispersionExp)
    let r = rate * duration * p / (1.0 - p)
    // Use Gamma-Poisson mixture (equivalent to NegativeBinomial, but O(1))
    // See issue for why we're not sampling from it directly: https://github.com/mathnet/mathnet-numerics/issues/320
    let lambda = Gamma(r, (1.0 - p) / p, rng).Sample()
    Poisson(lambda, rng).Sample()

// Test samples
let rng = Random(42)

printfn "Consolidation (rate=5, 60s):"
for _ in 1..10 do
    printfn "  %d" (sampleTradeCount rng 5.0 1.0 60.0)

printfn "\nStrongTrend (rate=50, 60s):"
for _ in 1..10 do
    printfn "  %d" (sampleTradeCount rng 50.0 1.0 60.0)