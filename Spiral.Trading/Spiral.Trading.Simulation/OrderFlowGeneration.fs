module Spiral.Trading.Simulation.OrderFlowGeneration

open System
open MathNet.Numerics.Distributions
open Spiral.Trading.Simulation.EpisodeMCMC

/// A single trade
type Trade = {
    Time: float      // Seconds from start of episode
    Trend: Trend
}

/// Parameters for order flow generation within a trend
type OrderFlowParams = {
    TradeRatePerSecond: float    // Mean trades per second
    DispersionExp: float         // 0 = Poisson-like, 1 = 2x variance, etc.
}

/// Get order flow parameters for a trend type
let getOrderFlowParams (trend: Trend) : OrderFlowParams =
    match trend with
    | StrongUptrend ->   { TradeRatePerSecond = 50.0; DispersionExp = 1.0 }
    | MidUptrend ->      { TradeRatePerSecond = 40.0; DispersionExp = 1.0 }
    | WeakUptrend ->     { TradeRatePerSecond = 25.0; DispersionExp = 1.0 }
    | Consolidation ->   { TradeRatePerSecond = 10.0; DispersionExp = 1.0 }
    | WeakDowntrend ->   { TradeRatePerSecond = 25.0; DispersionExp = 1.0 }
    | MidDowntrend ->    { TradeRatePerSecond = 40.0; DispersionExp = 1.0 }
    | StrongDowntrend -> { TradeRatePerSecond = 50.0; DispersionExp = 1.0 }

/// Sample trade count using Gamma-Poisson mixture (equivalent to NegativeBinomial)
let sampleTradeCount (rng: Random) (rate: float) (dispersionExp: float) (duration: float) =
    let p = Math.Pow(2.0, -dispersionExp)
    let r = rate * duration * p / (1.0 - p)
    // Use Gamma-Poisson mixture (equivalent to NegativeBinomial, but O(1))
    // See: https://github.com/mathnet/mathnet-numerics/issues/320
    let lambda = Gamma(r, (1.0 - p) / p, rng).Sample()
    Poisson(lambda, rng).Sample()

/// Generate uniformly distributed timestamps within an interval
let generateTimestamps (rng: Random) (startTime: float) (duration: float) (count: int) : float[] =
    let timestamps = Array.init count (fun _ -> startTime + rng.NextDouble() * duration)
    Array.sortInPlace timestamps
    timestamps

/// Generate trades for a single trend episode
let generateEpisodeTrades (rng: Random) (episode: Episode<Trend>) : Trade[] =
    let durationSeconds = episode.Duration * 60.0
    let params' = getOrderFlowParams episode.Label
    let tradeCount = sampleTradeCount rng params'.TradeRatePerSecond params'.DispersionExp durationSeconds
    let timestamps = generateTimestamps rng 0.0 durationSeconds tradeCount
    timestamps |> Array.map (fun ts -> { Time = ts; Trend = episode.Label })

/// Print summary statistics for generated trades
let printTradesSummary (trades: Trade[]) : unit =
    if trades.Length = 0 then
        printfn "No trades generated"
    else
        let duration = trades.[trades.Length - 1].Time
        let avgRate = float trades.Length / duration
        
        printfn "Trade Summary:"
        printfn "  Total trades: %d" trades.Length
        printfn "  Duration: %.1f seconds (%.1f minutes)" duration (duration / 60.0)
        printfn "  Average rate: %.2f trades/sec" avgRate
