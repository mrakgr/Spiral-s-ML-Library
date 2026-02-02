module Spiral.Trading.Simulation.OrderFlowGeneration

open System
open MathNet.Numerics.Distributions
open Spiral.Trading.Simulation.EpisodeMCMC

/// A single trade
type Trade = {
    Time: float      // Seconds from start of episode
    Price: float     // Execution price
    Size: int        // Number of shares/contracts
    Trend: Trend
}

/// Parameters for order flow generation within a trend
type OrderFlowParams = {
    TradeRatePerSecond: float    // Mean trades per second
    DispersionExp: float         // 0 = Poisson-like, 1 = 2x variance, etc.
}

/// Parameters for price generation (GBM)
type PriceParams = {
    DriftPerSecond: float        // Expected price change per second (as fraction)
    VolatilityPerSecond: float   // Std dev of price change per second (as fraction)
}

/// Parameters for trade size generation (Pareto)
type SizeParams = {
    MinSize: float               // Scale parameter (minimum trade size)
    Alpha: float                 // Shape parameter (lower = heavier tail)
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

/// Get price parameters for a trend type (drift and volatility as fractions)
let getPriceParams (trend: Trend) : PriceParams =
    match trend with
    | StrongUptrend ->   { DriftPerSecond = 30e-6;  VolatilityPerSecond = 100e-6 }
    | MidUptrend ->      { DriftPerSecond = 15e-6;  VolatilityPerSecond = 80e-6 }
    | WeakUptrend ->     { DriftPerSecond = 7e-6;   VolatilityPerSecond = 60e-6 }
    | Consolidation ->   { DriftPerSecond = 0.0;    VolatilityPerSecond = 40e-6 }
    | WeakDowntrend ->   { DriftPerSecond = -7e-6;  VolatilityPerSecond = 60e-6 }
    | MidDowntrend ->    { DriftPerSecond = -15e-6; VolatilityPerSecond = 80e-6 }
    | StrongDowntrend -> { DriftPerSecond = -30e-6; VolatilityPerSecond = 100e-6 }

/// Get size parameters for a trend type (Pareto distribution)
/// Stronger trends have lower alpha (heavier tail, larger average sizes)
let getSizeParams (trend: Trend) : SizeParams =
    match trend with
    | StrongUptrend ->   { MinSize = 1.0; Alpha = 1.5 }
    | MidUptrend ->      { MinSize = 1.0; Alpha = 1.8 }
    | WeakUptrend ->     { MinSize = 1.0; Alpha = 2.0 }
    | Consolidation ->   { MinSize = 1.0; Alpha = 2.5 }
    | WeakDowntrend ->   { MinSize = 1.0; Alpha = 2.0 }
    | MidDowntrend ->    { MinSize = 1.0; Alpha = 1.8 }
    | StrongDowntrend -> { MinSize = 1.0; Alpha = 1.5 }

/// Sample trade count using Gamma-Poisson mixture (equivalent to NegativeBinomial)
let sampleTradeCount (rng: Random) (rate: float) (dispersionExp: float) (duration: float) =
    let p = Math.Pow(2.0, -dispersionExp)
    let r = rate * duration * p / (1.0 - p)
    // Use Gamma-Poisson mixture (equivalent to NegativeBinomial, but O(1))
    // See: https://github.com/mathnet/mathnet-numerics/issues/320
    let lambda = Gamma(r, (1.0 - p) / p, rng).Sample()
    Poisson(lambda, rng).Sample()

/// Sample trade size from Pareto distribution
let sampleSize (rng: Random) (sizeParams: SizeParams) : int =
    let pareto = Pareto(sizeParams.MinSize, sizeParams.Alpha, rng)
    max 1 (int (pareto.Sample()))

/// Generate uniformly distributed timestamps within an interval
let generateTimestamps (rng: Random) (startTime: float) (duration: float) (count: int) : float[] =
    let timestamps = Array.init count (fun _ -> startTime + rng.NextDouble() * duration)
    Array.sortInPlace timestamps
    timestamps

/// Generate prices at trade timestamps using Geometric Brownian Motion
/// Returns array of prices and the final price for chaining
let generatePrices (rng: Random) (priceParams: PriceParams) (startPrice: float) (timestamps: float[]) : float[] * float =
    if timestamps.Length = 0 then
        [||], startPrice
    else
        let prices = Array.zeroCreate timestamps.Length
        let normal = Normal(0.0, 1.0, rng)
        let mutable price = startPrice
        let mutable prevTime = 0.0
        
        for i in 0 .. timestamps.Length - 1 do
            let dt = timestamps.[i] - prevTime
            let drift = priceParams.DriftPerSecond
            let vol = priceParams.VolatilityPerSecond
            // GBM: S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
            let z = normal.Sample()
            price <- price * exp((drift - vol * vol / 2.0) * dt + vol * sqrt(dt) * z)
            prices.[i] <- price
            prevTime <- timestamps.[i]
        
        prices, price

/// Generate trades for a single trend episode
/// Returns trades and the ending price for chaining to next episode
let generateEpisodeTrades (rng: Random) (startPrice: float) (episode: Episode<Trend>) : Trade[] * float =
    let durationSeconds = episode.Duration * 60.0
    let orderFlowParams = getOrderFlowParams episode.Label
    let priceParams = getPriceParams episode.Label
    let sizeParams = getSizeParams episode.Label
    
    let tradeCount = sampleTradeCount rng orderFlowParams.TradeRatePerSecond orderFlowParams.DispersionExp durationSeconds
    let timestamps = generateTimestamps rng 0.0 durationSeconds tradeCount
    let prices, endPrice = generatePrices rng priceParams startPrice timestamps
    
    let trades = Array.init tradeCount (fun i -> {
        Time = timestamps.[i]
        Price = prices.[i]
        Size = sampleSize rng sizeParams
        Trend = episode.Label
    })
    
    trades, endPrice

/// Print summary statistics for generated trades
let printTradesSummary (trades: Trade[]) : unit =
    if trades.Length = 0 then
        printfn "No trades generated"
    else
        let duration = trades.[trades.Length - 1].Time
        let avgRate = float trades.Length / duration
        let totalVolume = trades |> Array.sumBy (fun t -> t.Size)
        let avgSize = float totalVolume / float trades.Length
        let maxSize = trades |> Array.map (fun t -> t.Size) |> Array.max
        let startPrice = trades.[0].Price
        let endPrice = trades.[trades.Length - 1].Price
        let returnPct = (endPrice - startPrice) / startPrice * 100.0
        
        printfn "Trade Summary:"
        printfn "  Total trades: %d" trades.Length
        printfn "  Duration: %.1f seconds (%.1f minutes)" duration (duration / 60.0)
        printfn "  Average rate: %.2f trades/sec" avgRate
        printfn "  Total volume: %d" totalVolume
        printfn "  Avg size: %.1f, Max size: %d" avgSize maxSize
        printfn "  Start price: %.4f, End price: %.4f" startPrice endPrice
        printfn "  Return: %.2f%%" returnPct
