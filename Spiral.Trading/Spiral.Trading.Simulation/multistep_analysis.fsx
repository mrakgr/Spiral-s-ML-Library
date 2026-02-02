#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

// =============================================================================
// Multi-step GBM test - where does the bias come from?
// =============================================================================

let activitySigma = 1.0
let baseVol = 100e-6
let drift = 30e-6
let correction = exp(activitySigma * activitySigma / 8.0)
let startPrice = 100.0
let durationSeconds = 60.0

let sampleActivity (rng: Random) sigma =
    let mu = -sigma * sigma / 2.0
    LogNormal(mu, sigma, rng).Sample()

let generateTimestamps (rng: Random) duration count =
    let ts = Array.init count (fun _ -> rng.NextDouble() * duration)
    Array.sortInPlace ts
    ts

printfn "=== Multi-step GBM Analysis ==="
printfn ""

let runs = 500000
let tradesPerRun = 100

let rng = Random(42)
let normal = Normal(0.0, 1.0, rng)

let mutable sumFinalPrice = 0.0

for _ in 1 .. runs do
    let timestamps = generateTimestamps rng durationSeconds tradesPerRun
    let mutable price = startPrice
    let mutable prevTime = 0.0
    
    for t in timestamps do
        let dt = t - prevTime
        let activity = sampleActivity rng activitySigma
        let scaledVol = baseVol * correction * sqrt(activity)
        let z = normal.Sample()
        price <- price * exp((drift - scaledVol * scaledVol / 2.0) * dt + scaledVol * sqrt(dt) * z)
        prevTime <- t
    
    sumFinalPrice <- sumFinalPrice + price

let avgFinalPrice = sumFinalPrice / float runs
let expectedFinalPrice = startPrice * exp(drift * durationSeconds)

printfn "Average final price: %.6f" avgFinalPrice
printfn "Expected final price: %.6f" expectedFinalPrice
printfn "Ratio: %.6f" (avgFinalPrice / expectedFinalPrice)
printfn ""
printfn "Average return: %.6f%%" ((avgFinalPrice / startPrice - 1.0) * 100.0)
printfn "Expected return (exp(drift*T)-1): %.6f%%" ((exp(drift * durationSeconds) - 1.0) * 100.0)
