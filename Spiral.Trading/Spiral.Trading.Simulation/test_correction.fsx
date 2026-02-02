#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// Test: Verify that correction factor makes E[|price_change|] = base_vol * sqrt(2/pi)
//
// Model:
//   activity ~ LogNormal(mu, sigma) with E[activity] = 1 (mu = -sigma^2/2)
//   size = base_size * activity
//   price_change ~ N(0, corrected_vol * sqrt(activity))
//   
// Where:
//   correction = exp(-sigma^2 * p * (p-1) / 2) for p = 0.5
//              = exp(sigma^2 / 8)
//   corrected_vol = base_vol * correction
// =============================================================================

let baseVol = 0.001
let baseSize = 100.0
let p = 0.5  // sqrt law

printfn "=== Testing Correction Factor ==="
printfn ""

for sigma in [0.5; 1.0; 1.5; 2.0] do
    let mu = -sigma * sigma / 2.0
    let correction = exp(sigma * sigma / 8.0)  // = exp(-sigma^2 * 0.5 * (-0.5) / 2)
    let correctedVol = baseVol * correction
    
    // Sample
    let samples = Array.init 50000 (fun _ ->
        let activity = LogNormal(mu, sigma, rng).Sample()
        let size = baseSize * activity
        let vol = correctedVol * sqrt(activity)
        let priceChange = Normal(0.0, vol, rng).Sample()
        (activity, size, priceChange))
    
    let avgAbsPriceChange = samples |> Array.averageBy (fun (_, _, dp) -> abs dp)
    let avgSize = samples |> Array.averageBy (fun (_, s, _) -> s)
    
    // Expected E[|N(0, sigma)|] = sigma * sqrt(2/pi)
    let expectedAbsChange = baseVol * sqrt(2.0 / Math.PI)
    let ratio = avgAbsPriceChange / expectedAbsChange
    
    printfn "sigma = %.1f:" sigma
    printfn "  correction factor = %.4f" correction
    printfn "  E[size] = %.1f (expected: %.1f)" avgSize baseSize
    printfn "  E[|dP|] = %.6f (expected: %.6f)" avgAbsPriceChange expectedAbsChange
    printfn "  ratio   = %.4f (should be ~1.0)" ratio
    printfn ""
