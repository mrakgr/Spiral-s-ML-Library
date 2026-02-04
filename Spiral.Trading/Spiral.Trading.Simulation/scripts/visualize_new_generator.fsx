#r "nuget: MathNet.Numerics, 5.0.0"
#r "nuget: MathNet.Numerics.FSharp, 5.0.0"

#load "../EpisodeMCMC.fs"
#load "../OrderFlowGeneration.fs"

open System
open Spiral.Trading.Simulation.EpisodeMCMC
open Spiral.Trading.Simulation.OrderFlowGeneration

let rng = Random(42)

// Generate a day structure (same as old generator)
let dayResult = generateDay SessionLevel.defaultConfig TrendLevel.defaultConfig MCMC.defaultConfig rng 390.0

printDayResult dayResult

// Generate trades for all episodes
let startPrice = 100.0
let mutable price = startPrice
let mutable timeOffset = 0.0
let allTrades = ResizeArray<Trade>()

for sessionIdx in 0 .. dayResult.Sessions.Length - 1 do
    for episode in dayResult.Trends.[sessionIdx] do
        let trades, endPrice = generateEpisodeTrades rng price episode
        // Adjust timestamps to be relative to day start
        for trade in trades do
            allTrades.Add({ trade with Time = trade.Time + timeOffset })
        timeOffset <- timeOffset + episode.Duration * 60.0
        price <- endPrice

let trades = allTrades.ToArray()
printfn "\nTotal trades: %d" trades.Length
printfn "Time range: %.1f - %.1f seconds" trades.[0].Time trades.[trades.Length-1].Time

// Aggregate to 1-second OHLC bars
let totalSeconds = 390 * 60
let bars = Array.init totalSeconds (fun sec ->
    let secTrades = trades |> Array.filter (fun t -> int t.Time = sec)
    if secTrades.Length > 0 then
        Some {|
            Time = sec
            Open = secTrades.[0].Price
            High = secTrades |> Array.map (fun t -> t.Price) |> Array.max
            Low = secTrades |> Array.map (fun t -> t.Price) |> Array.min
            Close = secTrades.[secTrades.Length-1].Price
            Volume = secTrades |> Array.sumBy (fun t -> t.Size)
            Trend = secTrades.[0].Trend
            TradeCount = secTrades.Length
        |}
    else
        None
)

// Fill gaps with previous close (forward fill)
let filledBars = Array.zeroCreate totalSeconds
let mutable lastBar = None
for i in 0 .. totalSeconds - 1 do
    match bars.[i], lastBar with
    | Some bar, _ -> 
        filledBars.[i] <- bar
        lastBar <- Some bar
    | None, Some prev ->
        filledBars.[i] <- {| prev with Time = i; TradeCount = 0; Volume = 0 |}
    | None, None ->
        filledBars.[i] <- {| Time = i; Open = startPrice; High = startPrice; Low = startPrice; Close = startPrice; Volume = 0; Trend = Consolidation; TradeCount = 0 |}

// Export to CSV
let outputPath = "/home/mrakgr/Spiral-s-ML-Library/Spiral.Trading/data/new_generator_bars.csv"
use writer = new IO.StreamWriter(outputPath)
writer.WriteLine("Time,Open,High,Low,Close,Volume,Trend,TradeCount")
for bar in filledBars do
    writer.WriteLine(sprintf "%d,%.6f,%.6f,%.6f,%.6f,%d,%A,%d" 
        bar.Time bar.Open bar.High bar.Low bar.Close bar.Volume bar.Trend bar.TradeCount)
printfn "\nExported %d bars to %s" filledBars.Length outputPath

// Summary
let barsWithTrades = bars |> Array.choose id
printfn "Bars with trades: %d / %d" barsWithTrades.Length totalSeconds
let avgTradesPerBar = barsWithTrades |> Array.averageBy (fun b -> float b.TradeCount)
printfn "Avg trades per bar (when > 0): %.1f" avgTradesPerBar
