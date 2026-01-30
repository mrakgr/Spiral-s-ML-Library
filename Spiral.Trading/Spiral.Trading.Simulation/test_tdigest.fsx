#r "nuget: T-Digest.NET"

open System
open TDigestNet

let rng = Random(42)

// Create a t-digest with compression factor 1024
let td = TDigest(1024.0)

// Add 10000 random normal values
printfn "Adding 10000 random values..."
for _ in 1..10000 do
    // Box-Muller transform for normal distribution
    let u1 = rng.NextDouble()
    let u2 = rng.NextDouble()
    let z = sqrt(-2.0 * log(u1)) * cos(2.0 * Math.PI * u2)
    td.Add(z)

printfn "Count: %d" (int td.Count)

// Test CDF at various points
printfn "\nCDF values:"
printfn "  CDF(-2) = %.4f (expected ~0.023)" (td.Cdf(-2.0))
printfn "  CDF(-1) = %.4f (expected ~0.159)" (td.Cdf(-1.0))
printfn "  CDF(0)  = %.4f (expected ~0.500)" (td.Cdf(0.0))
printfn "  CDF(1)  = %.4f (expected ~0.841)" (td.Cdf(1.0))
printfn "  CDF(2)  = %.4f (expected ~0.977)" (td.Cdf(2.0))

// Test quantiles (inverse CDF)
printfn "\nQuantiles:"
printfn "  Q(0.01) = %.4f (expected ~-2.33)" (td.Quantile(0.01))
printfn "  Q(0.50) = %.4f (expected ~0.00)" (td.Quantile(0.50))
printfn "  Q(0.99) = %.4f (expected ~2.33)" (td.Quantile(0.99))

// Normalize function: CDF * 2 - 1 maps to [-1, 1]
let normalize (td: TDigest) (x: float) = td.Cdf(x) * 2.0 - 1.0

printfn "\nNormalized values (should be in [-1, 1]):"
printfn "  normalize(-2) = %.4f" (normalize td -2.0)
printfn "  normalize(0)  = %.4f" (normalize td 0.0)
printfn "  normalize(2)  = %.4f" (normalize td 2.0)
