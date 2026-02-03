#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

// =============================================================================
// The real issue: E[exp(-scaledVol²/2 * dt + scaledVol * sqrt(dt) * Z)]
//
// For fixed vol: E_Z[exp(-vol²/2 * dt + vol * sqrt(dt) * Z)] = 1 (by design)
//
// For stochastic vol, we need:
//   E_activity[ E_Z[exp(-scaledVol²/2 * dt + scaledVol * sqrt(dt) * Z)] ]
// = E_activity[ 1 ]  (inner expectation is 1 for any fixed scaledVol)
// = 1
//
// So the scaledVol² correction IS correct! The issue must be elsewhere.
// =============================================================================

printfn "=== Verifying Stochastic Vol Correction ==="
printfn ""

let rng = Random(42)
let activitySigma = 1.0
let baseVol = 100e-6
let correction = exp(activitySigma * activitySigma / 8.0)
let dt = 0.6  // 60 seconds / 100 trades

let sampleActivity sigma =
    let mu = -sigma * sigma / 2.0
    LogNormal(mu, sigma, rng).Sample()

// Test: E[exp(-scaledVol²/2 * dt + scaledVol * sqrt(dt) * Z)] should be 1
let samples = 500000
let normal = Normal(0.0, 1.0, rng)

let mutable sum = 0.0
for _ in 1 .. samples do
    let activity = sampleActivity activitySigma
    let scaledVol = baseVol * correction * sqrt(activity)
    let z = normal.Sample()
    let factor = exp(-scaledVol * scaledVol / 2.0 * dt + scaledVol * sqrt(dt) * z)
    sum <- sum + factor

printfn "E[exp(-scaledVol²/2 * dt + scaledVol * sqrt(dt) * Z)] = %.6f (should be 1.0)" (sum / float samples)
printfn ""

// Now test the full GBM step
printfn "=== Full GBM Step Test ==="
let drift = 30e-6
let startPrice = 100.0

let mutable sumPrice = 0.0
for _ in 1 .. samples do
    let activity = sampleActivity activitySigma
    let scaledVol = baseVol * correction * sqrt(activity)
    let z = normal.Sample()
    let price = startPrice * exp((drift - scaledVol * scaledVol / 2.0) * dt + scaledVol * sqrt(dt) * z)
    sumPrice <- sumPrice + price

let expectedPrice = startPrice * exp(drift * dt)
printfn "E[price] = %.6f" (sumPrice / float samples)
printfn "Expected (S0 * exp(drift*dt)) = %.6f" expectedPrice
printfn "Ratio = %.6f" ((sumPrice / float samples) / expectedPrice)
