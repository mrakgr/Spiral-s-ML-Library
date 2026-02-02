#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// Option 4: Latent Activity Variable
// 
// The idea: a hidden "activity" drives both size and volatility
//   activity ~ Gamma (or other positive distribution)
//   size = activity * base_size
//   price_change ~ N(0, base_vol * sqrt(activity))
//
// This is equivalent to Option 3 (conditional sampling) where:
//   activity = size / base_size
// =============================================================================

let baseSize = 1.0
let baseVol = 0.001

// Gamma parameters for activity (shape=alpha, rate=beta)
// Mean = alpha/beta, Variance = alpha/beta^2
let activityShape = 1.5  // Same as Pareto alpha for comparison
let activityRate = 1.5   // Mean activity = 1.0

printfn "=== Option 4: Latent Activity Variable ==="
printfn "activity ~ Gamma(shape=%.1f, rate=%.1f)" activityShape activityRate
printfn "size = activity * base_size"
printfn "price_change ~ N(0, base_vol * sqrt(activity))"
printfn ""

let sampleWithLatentActivity () =
    let activity = Gamma(activityShape, 1.0 / activityRate, rng).Sample()
    let size = activity * baseSize
    let vol = baseVol * sqrt(activity)
    let priceChange = Normal(0.0, vol, rng).Sample()
    (activity, size, priceChange)

let samples = Array.init 10000 (fun _ -> sampleWithLatentActivity())

// Verify the sqrt relationship holds
let bins = [| 0.0; 0.5; 1.0; 2.0; 4.0; 8.0; 16.0; infinity |]

printfn "Activity Range    | Count | E[|dP|]   | baseVol*sqrt(mid) | Ratio"
printfn "------------------|-------|-----------|-------------------|------"

for i in 0 .. bins.Length - 2 do
    let lo, hi = bins.[i], bins.[i+1]
    let inBin = samples |> Array.filter (fun (a, _, _) -> a >= lo && a < hi)
    if inBin.Length > 10 then
        let avgAbsChange = inBin |> Array.averageBy (fun (_, _, dp) -> abs dp)
        let midpoint = if hi = infinity then lo * 2.0 else (lo + hi) / 2.0
        let expected = baseVol * sqrt(midpoint) * sqrt(2.0 / Math.PI)
        let ratio = avgAbsChange / expected
        printfn "[%5.1f, %5.1f)    | %5d | %.6f | %.6f          | %.3f" 
            lo hi inBin.Length avgAbsChange expected ratio

printfn ""
printfn "=== Equivalence with Option 3 ==="
printfn "If we set activity = size / base_size, then:"
printfn "  price_change ~ N(0, base_vol * sqrt(size / base_size))"
printfn "  price_change ~ N(0, base_vol / sqrt(base_size) * sqrt(size))"
printfn ""
printfn "Same sqrt(size) relationship, just different parameterization."
