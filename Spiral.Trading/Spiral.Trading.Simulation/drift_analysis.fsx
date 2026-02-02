#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

// =============================================================================
// Investigating the drift bias
//
// For GBM: S(T) = S(0) * exp((mu - sigma²/2)*T + sigma*W(T))
// 
// The expected value is: E[S(T)] = S(0) * exp(mu * T)
// So: E[S(T)/S(0)] = exp(mu * T)
// And: E[S(T)/S(0) - 1] = exp(mu * T) - 1  (not mu * T!)
//
// For small mu*T: exp(mu*T) - 1 ≈ mu*T + (mu*T)²/2 + ...
// =============================================================================

let priceParams = {| DriftPerSecond = 30e-6; VolatilityPerSecond = 100e-6 |}
let activitySigma = 1.0
let startPrice = 100.0
let durationSeconds = 60.0

let getCorrection sigma = exp(sigma * sigma / 8.0)
let sampleActivity (rng: Random) sigma =
    let mu = -sigma * sigma / 2.0
    LogNormal(mu, sigma, rng).Sample()

let generateTimestamps (rng: Random) duration count =
    let ts = Array.init count (fun _ -> rng.NextDouble() * duration)
    Array.sortInPlace ts
    ts

printfn "=== Expected Return Analysis ==="
printfn ""
printfn "drift = %.6f per second" priceParams.DriftPerSecond
printfn "duration = %.0f seconds" durationSeconds
printfn ""

let driftT = priceParams.DriftPerSecond * durationSeconds
printfn "drift * T = %.6f" driftT
printfn "exp(drift * T) - 1 = %.6f" (exp(driftT) - 1.0)
printfn "Difference: %.9f" (exp(driftT) - 1.0 - driftT)
printfn ""
printfn "So the correct expected arithmetic return is exp(drift*T) - 1, not drift*T"
