#r "nuget: MathNet.Numerics"
#load "EpisodeMCMC.fs"
#load "OrderFlowGeneration.fs"

open System
open Spiral.Trading.Simulation.EpisodeMCMC
open Spiral.Trading.Simulation.OrderFlowGeneration

let rng = Random(42)
let startPrice = 100.0

printfn "=== Testing Correlated Price-Volume Model ==="
printfn ""

let episode = { Label = StrongUptrend; Duration = 1.0 }
let trades, endPrice = generateEpisodeTrades rng startPrice episode

printfn "Episode: %A, Duration: %.0f min" episode.Label episode.Duration
printfn "Trades: %d" trades.Length
printfn "Start: %.4f, End: %.4f, Return: %.2f%%" startPrice endPrice ((endPrice - startPrice) / startPrice * 100.0)
printfn ""

let sizes = trades |> Array.map (fun t -> float t.Size)
let sortedSizes = Array.sort sizes
let medianSize = sortedSizes.[sortedSizes.Length / 2]
printfn "Size stats: Median=%.1f, Mean=%.1f, Max=%.0f" medianSize (Array.average sizes) (Array.max sizes)
printfn ""

// Check correlation
if trades.Length > 1 then
    let priceChanges = Array.init (trades.Length - 1) (fun i -> abs (trades.[i+1].Price - trades.[i].Price))
    let sizesForCorr = trades.[1..] |> Array.map (fun t -> float t.Size)
    
    let meanSize = Array.average sizesForCorr
    let meanChange = Array.average priceChanges
    let cov = Array.map2 (fun s c -> (s - meanSize) * (c - meanChange)) sizesForCorr priceChanges |> Array.average
    let stdSize = sqrt (Array.averageBy (fun s -> (s - meanSize) ** 2.0) sizesForCorr)
    let stdChange = sqrt (Array.averageBy (fun c -> (c - meanChange) ** 2.0) priceChanges)
    let corr = cov / (stdSize * stdChange)
    
    printfn "Correlation(size, |price_change|): %.3f" corr
