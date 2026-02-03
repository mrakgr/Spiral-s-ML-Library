#r "nuget: MathNet.Numerics"
#load "EpisodeMCMC.fs"
#load "OrderFlowGeneration.fs"

open System
open Spiral.Trading.Simulation.OrderFlowGeneration

let rng = Random(42)
let rate = 50.0
let duration = 60.0
let expectedMean = rate * duration

printfn "Expected mean: %.0f trades" expectedMean
printfn ""

for dispExp in [0.0; 1.0; 2.0; 3.0] do
    let samples = Array.init 1000 (fun _ -> sampleTradeCount rng rate dispExp duration)
    let mean = Array.averageBy float samples
    let variance = samples |> Array.averageBy (fun x -> (float x - mean) ** 2.0)
    printfn "dispExp=%.1f: mean=%.0f, var=%.0f, std=%.0f" dispExp mean variance (sqrt variance)
