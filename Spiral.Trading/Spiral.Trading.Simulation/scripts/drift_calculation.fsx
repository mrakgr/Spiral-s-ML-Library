#r "nuget: MathNet.Numerics, 5.0.0"

#load "../EpisodeMCMC.fs"
#load "../OrderFlowGeneration.fs"

open System
open Spiral.Trading.Simulation.OrderFlowGeneration

// With 0.1% drift per sqrt(1M) shares
// If we trade 100M shares/day, sqrt(100M) = 10,000
// Expected drift = 0.001 * 10,000 / 1000 = 0.01 = 1%? Let's verify

let driftPerSqrtMillion = 0.001  // 0.1%
let dailyVolume = 100_000_000.0  // 100M shares
let sqrtDailyVol = sqrt(dailyVolume)

printfn "Drift per sqrt(1M): %.4f%%" (driftPerSqrtMillion * 100.0)
printfn "Daily volume: %.0f shares" dailyVolume
printfn "sqrt(daily volume): %.0f" sqrtDailyVol
printfn ""

// Expected daily drift = driftPerSqrtShare * sqrt(totalVolume)
let driftPerSqrtShare = driftPerSqrtMillion / sqrt(1_000_000.0)
let expectedDailyDrift = driftPerSqrtShare * sqrtDailyVol
printfn "Drift per sqrt-share: %.10f" driftPerSqrtShare
printfn "Expected daily drift: %.4f%%" (expectedDailyDrift * 100.0)
printfn ""

// What about with our actual simulation?
// ~600k trades/day, avg size ~200 = 120M shares
let actualDailyVolume = 600_000.0 * 200.0
let sqrtActualVol = sqrt(actualDailyVolume)
let actualExpectedDrift = driftPerSqrtShare * sqrtActualVol
printfn "Actual daily volume (600k trades × 200): %.0f shares" actualDailyVolume
printfn "sqrt(actual volume): %.0f" sqrtActualVol
printfn "Expected daily drift: %.4f%%" (actualExpectedDrift * 100.0)
