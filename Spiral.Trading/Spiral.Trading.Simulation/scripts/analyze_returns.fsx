#r "nuget: MathNet.Numerics, 5.0.0"

#load "../EpisodeMCMC.fs"
#load "../OrderFlowGeneration.fs"

open System
open Spiral.Trading.Simulation.EpisodeMCMC
open Spiral.Trading.Simulation.OrderFlowGeneration

let rng = Random(42)

// Generate a single StrongDowntrend episode (5 minutes)
let episode = { Label = StrongDowntrend; Duration = 5.0 }
let trades, _ = generateEpisodeTrades rng 100.0 episode

// Calculate per-trade returns
let returns = 
    trades 
    |> Array.pairwise 
    |> Array.map (fun (t1, t2) -> (t2.Price - t1.Price) / t1.Price * 100.0)

printfn "StrongDowntrend episode (5 min):"
printfn "  Trades: %d" trades.Length
printfn "  Avg size: %.1f" (trades |> Array.averageBy (fun t -> float t.Size))
printfn ""
printfn "Per-trade return stats:"
printfn "  Mean: %.6f%%" (returns |> Array.average)
printfn "  StdDev: %.6f%%" (let m = Array.average returns in returns |> Array.map (fun r -> (r-m)*(r-m)) |> Array.average |> sqrt)
printfn "  Min: %.6f%%" (returns |> Array.min)
printfn "  Max: %.6f%%" (returns |> Array.max)
printfn ""
printfn "Total return: %.4f%%" ((trades.[trades.Length-1].Price - trades.[0].Price) / trades.[0].Price * 100.0)
printfn ""

// Show first 20 trades
printfn "First 20 trades:"
printfn "Trade,Price,Size,Return%%"
printfn "0,%.6f,%d," trades.[0].Price trades.[0].Size
for i in 1 .. min 19 (trades.Length - 1) do
    let ret = (trades.[i].Price - trades.[i-1].Price) / trades.[i-1].Price * 100.0
    printfn "%d,%.6f,%d,%.6f" i trades.[i].Price trades.[i].Size ret
