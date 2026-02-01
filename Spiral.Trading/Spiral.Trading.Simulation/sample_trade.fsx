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
    let nb = NegativeBinomial(r, p, rng)
    nb.Sample()