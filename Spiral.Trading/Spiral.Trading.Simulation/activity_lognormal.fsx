#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// LogNormal with E[activity] = 1
//
// For LogNormal(mu, sigma):
//   E[X] = exp(mu + sigma^2/2)
//   E[X^p] = exp(p*mu + p^2*sigma^2/2)
//
// To get E[X] = 1:
//   mu + sigma^2/2 = 0
//   mu = -sigma^2/2
//
// Then for power p:
//   E[X^p] = exp(p*(-sigma^2/2) + p^2*sigma^2/2)
//          = exp(sigma^2 * (p^2 - p) / 2)
//          = exp(sigma^2 * p * (p - 1) / 2)
//
// Scale correction factor: 1 / E[X^p] = exp(-sigma^2 * p * (p - 1) / 2)
// =============================================================================

printfn "=== LogNormal Activity with E[activity] = 1 ==="
printfn ""
printfn "For LogNormal(mu, sigma) with E[X] = 1, mu = -sigma^2/2"
printfn "E[X^p] = exp(sigma^2 * p * (p-1) / 2)"
printfn ""

let powers = [0.5; 0.6; 1.0]

printfn "sigma | E[X^0.5] | E[X^0.6] | Correction(0.5) | Correction(0.6)"
printfn "------|----------|----------|-----------------|----------------"

for sigma in [0.5; 1.0; 1.5; 2.0] do
    let expectedPower p = exp(sigma * sigma * p * (p - 1.0) / 2.0)
    let correction p = 1.0 / expectedPower p
    printfn " %.1f  |  %.4f   |  %.4f   |     %.4f      |     %.4f" 
        sigma (expectedPower 0.5) (expectedPower 0.6) (correction 0.5) (correction 0.6)

printfn ""
printfn "=== Empirical Verification (sigma = 1.0) ==="
let sigma = 1.0
let mu = -sigma * sigma / 2.0
let samples = Array.init 50000 (fun _ -> LogNormal(mu, sigma, rng).Sample())

printfn ""
printfn "E[activity] = %.4f (should be 1.0)" (Array.average samples)
printfn "E[X^0.5]    = %.4f (theoretical: %.4f)" 
    (samples |> Array.averageBy (fun x -> x ** 0.5)) 
    (exp(sigma * sigma * 0.5 * (0.5 - 1.0) / 2.0))
printfn "E[X^0.6]    = %.4f (theoretical: %.4f)" 
    (samples |> Array.averageBy (fun x -> x ** 0.6)) 
    (exp(sigma * sigma * 0.6 * (0.6 - 1.0) / 2.0))

printfn ""
printfn "=== Interpretation ==="
printfn "With sigma=1.0:"
printfn "  sqrt law (p=0.5): need to multiply base_vol by %.3f" (1.0 / exp(1.0 * 0.5 * (-0.5) / 2.0))
printfn "  3/5 law  (p=0.6): need to multiply base_vol by %.3f" (1.0 / exp(1.0 * 0.6 * (-0.4) / 2.0))
printfn ""
printfn "The 3/5 law requires less correction because 0.6 is closer to 1.0"
