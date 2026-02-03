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

/// Parameters for trade size generation (LogNormal activity model)
type ActivityParams = {
    BaseSize: float              // Mean trade size (E[size] = BaseSize)
    Sigma: float                 // LogNormal sigma (dispersion of activity)
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

/// Get activity parameters for a trend type (LogNormal activity model)
/// Stronger trends have higher sigma (more variable activity, larger occasional trades)
let getActivityParams (trend: Trend) : ActivityParams =
    match trend with
    | StrongUptrend ->   { BaseSize = 100.0; Sigma = 1.2 }
    | MidUptrend ->      { BaseSize = 100.0; Sigma = 1.0 }
    | WeakUptrend ->     { BaseSize = 100.0; Sigma = 0.8 }
    | Consolidation ->   { BaseSize = 100.0; Sigma = 0.6 }
    | WeakDowntrend ->   { BaseSize = 100.0; Sigma = 0.8 }
    | MidDowntrend ->    { BaseSize = 100.0; Sigma = 1.0 }
    | StrongDowntrend -> { BaseSize = 100.0; Sigma = 1.2 }

/// Sample trade count using Gamma-Poisson mixture (equivalent to NegativeBinomial)
let sampleTradeCount (rng: Random) (rate: float) (dispersionExp: float) (duration: float) =
    let p = Math.Pow(2.0, -dispersionExp)
    let r = rate * duration * p / (1.0 - p)
    // Use Gamma-Poisson mixture (equivalent to NegativeBinomial, but O(1))
    // See: https://github.com/mathnet/mathnet-numerics/issues/320
    let lambda = Gamma(r, (1.0 - p) / p, rng).Sample()
    Poisson(lambda, rng).Sample()

/// Stochastic rounding: rounds up or down probabilistically based on fractional part
let stochasticRound (rng: Random) (x: float) : int =
    let floor = Math.Floor(x)
    let frac = x - floor
    int (if rng.NextDouble() < frac then floor + 1.0 else floor)

/// Correction factor so E[correction * sqrt(activity)] = 1
let getVolatilityCorrection (sigma: float) : float =
    exp(sigma * sigma / 8.0)

/// Sample activity from LogNormal with E[activity] = 1
let sampleActivity (rng: Random) (sigma: float) : float =
    let mu = -sigma * sigma / 2.0
    LogNormal(mu, sigma, rng).Sample()

/// Sample trade size from activity
let sampleSize (rng: Random) (baseSize: float) (activity: float) : int =
    max 1 (stochasticRound rng (baseSize * activity))

/// Generate uniformly distributed timestamps within an interval
let generateTimestamps (rng: Random) (startTime: float) (duration: float) (count: int) : float[] =
    let timestamps = Array.init count (fun _ -> startTime + rng.NextDouble() * duration)
    Array.sortInPlace timestamps
    timestamps

/// Generate correlated prices and sizes at trade timestamps using GBM with activity scaling
/// Returns array of (price, size) pairs and the final price for chaining
let generatePricesAndSizes 
    (rng: Random) 
    (priceParams: PriceParams) 
    (activityParams: ActivityParams)
    (startPrice: float) 
    (timestamps: float[]) 
    : (float * int)[] * float =
    
    if timestamps.Length = 0 then
        [||], startPrice
    else
        let correction = getVolatilityCorrection activityParams.Sigma
        let normal = Normal(0.0, 1.0, rng)
        let results = Array.zeroCreate timestamps.Length
        let mutable price = startPrice
        let mutable prevTime = 0.0
        
        for i in 0 .. timestamps.Length - 1 do
            let dt = timestamps.[i] - prevTime
            let activity = sampleActivity rng activityParams.Sigma
            let size = sampleSize rng activityParams.BaseSize activity
            
            let drift = priceParams.DriftPerSecond
            let baseVol = priceParams.VolatilityPerSecond
            let scaledVol = baseVol * correction * sqrt(activity)
            
            // GBM with activity-scaled volatility
            let z = normal.Sample()
            price <- price * exp((drift - scaledVol * scaledVol / 2.0) * dt + scaledVol * sqrt(dt) * z)
            results.[i] <- (price, size)
            prevTime <- timestamps.[i]
        
        results, price

/// Generate trades for a single trend episode
/// Returns trades and the ending price for chaining to next episode
let generateEpisodeTrades (rng: Random) (startPrice: float) (episode: Episode<Trend>) : Trade[] * float =
    let durationSeconds = episode.Duration * 60.0
    let orderFlowParams = getOrderFlowParams episode.Label
    let priceParams = getPriceParams episode.Label
    let activityParams = getActivityParams episode.Label
    
    let tradeCount = sampleTradeCount rng orderFlowParams.TradeRatePerSecond orderFlowParams.DispersionExp durationSeconds
    let timestamps = generateTimestamps rng 0.0 durationSeconds tradeCount
    let pricesAndSizes, endPrice = generatePricesAndSizes rng priceParams activityParams startPrice timestamps
    
    let trades = Array.init tradeCount (fun i -> 
        let price, size = pricesAndSizes.[i]
        {
            Time = timestamps.[i]
            Price = price
            Size = size
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
