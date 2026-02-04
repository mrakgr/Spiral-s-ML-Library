#r "nuget: MathNet.Numerics, 5.0.0"
#r "nuget: MathNet.Numerics.FSharp, 5.0.0"

#load "../EpisodeMCMC.fs"
#load "../PriceGeneration.fs"

open System
open Spiral.Trading.Simulation.EpisodeMCMC
open Spiral.Trading.Simulation.PriceGeneration

let rng = Random(42)

// Generate a day structure
let dayResult = generateDay SessionLevel.defaultConfig TrendLevel.defaultConfig MCMC.defaultConfig rng 390.0

printDayResult dayResult

// Generate price bars
let startPrice = 100.0
let bars = generateDayBars rng startPrice dayResult

printBarsSummary bars

// Export to CSV for visualization
let outputPath = "/home/mrakgr/Spiral-s-ML-Library/Spiral.Trading/data/old_generator_bars.csv"
exportToCsv outputPath bars

// Also create a simple ASCII visualization of the price movement
printfn "\nPrice movement (sampled every 5 minutes):"
let sampleInterval = 300 // 5 minutes
for i in 0 .. sampleInterval .. bars.Length - 1 do
    let bar = bars.[i]
    let minutes = int bar.Time / 60
    let pctChange = (bar.Close - startPrice) / startPrice * 100.0
    let barLen = int (pctChange * 10.0) |> max -40 |> min 40
    let barStr = 
        if barLen >= 0 then String.replicate barLen "#"
        else String.replicate (-barLen) "-"
    printfn "%3d min | %6.2f%% | %s %A" minutes pctChange barStr bar.Trend
