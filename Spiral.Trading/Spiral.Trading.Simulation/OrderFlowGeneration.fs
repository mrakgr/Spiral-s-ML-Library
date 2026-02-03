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

/// Parameters for order flow generation within a trend (LogNormal model)
type OrderFlowParams = {
    MedianTradesPerSecond: float  // Typical trade rate (50th percentile)
    MeanTradesPerSecond: float    // Average trade rate (>= Median due to right skew)
}

/// Parameters for price generation (GBM)
type PriceParams = {
    DriftPerSecond: float        // Expected price change per second (as fraction)
    VolatilityPerSecond: float   // Std dev of price change per second (as fraction)
}

/// Parameters for trade size generation (LogNormal activity model)
/// Parameterized by median and mean for intuitive interpretation
type ActivityParams = {
    MedianSize: float            // Typical trade size (50th percentile)
    MeanSize: float              // Average trade size (>= MedianSize due to right skew)
}

/// Get order flow parameters for a trend type
let getOrderFlowParams (trend: Trend) : OrderFlowParams =
    match trend with
    | StrongUptrend ->   { MedianTradesPerSecond = 50.0; MeanTradesPerSecond = 60.0 }
    | MidUptrend ->      { MedianTradesPerSecond = 40.0; MeanTradesPerSecond = 48.0 }
    | WeakUptrend ->     { MedianTradesPerSecond = 25.0; MeanTradesPerSecond = 28.0 }
    | Consolidation ->   { MedianTradesPerSecond = 10.0; MeanTradesPerSecond = 11.0 }
    | WeakDowntrend ->   { MedianTradesPerSecond = 25.0; MeanTradesPerSecond = 28.0 }
    | MidDowntrend ->    { MedianTradesPerSecond = 40.0; MeanTradesPerSecond = 48.0 }
    | StrongDowntrend -> { MedianTradesPerSecond = 50.0; MeanTradesPerSecond = 60.0 }

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
/// Stronger trends have higher mean/median ratio (more large trades)
let getActivityParams (trend: Trend) : ActivityParams =
    match trend with
    | StrongUptrend ->   { MedianSize = 100.0; MeanSize = 200.0 }
    | MidUptrend ->      { MedianSize = 100.0; MeanSize = 150.0 }
    | WeakUptrend ->     { MedianSize = 100.0; MeanSize = 120.0 }
    | Consolidation ->   { MedianSize = 100.0; MeanSize = 110.0 }
    | WeakDowntrend ->   { MedianSize = 100.0; MeanSize = 120.0 }
    | MidDowntrend ->    { MedianSize = 100.0; MeanSize = 150.0 }
    | StrongDowntrend -> { MedianSize = 100.0; MeanSize = 200.0 }

/// Stochastic rounding: rounds up or down probabilistically based on fractional part
let stochasticRound (rng: Random) (x: float) : int =
    let floor = Math.Floor(x)
    let frac = x - floor
    int (if rng.NextDouble() < frac then floor + 1.0 else floor)

/// Sample trade count using LogNormal distribution
let sampleTradeCount (rng: Random) (orderFlowParams: OrderFlowParams) (duration: float) =
    let mu = log(orderFlowParams.MedianTradesPerSecond)
    let sigma = sqrt(2.0 * log(orderFlowParams.MeanTradesPerSecond / orderFlowParams.MedianTradesPerSecond))
    let count = LogNormal(mu, sigma, rng).Sample() * duration
    max 1 (stochasticRound rng count)

/// Convert median/mean parameterization to LogNormal mu/sigma
let activityMuSigma (activityParams: ActivityParams) : float * float =
    let mu = log(activityParams.MedianSize)
    let sigma = sqrt(2.0 * log(activityParams.MeanSize / activityParams.MedianSize))
    (mu, sigma)

/// Correction factor so E[correction * sqrt(activity)] = 1
let getVolatilityCorrection (sigma: float) : float =
    exp(sigma * sigma / 8.0)

/// Sample activity from LogNormal (returns raw size, not normalized)
let sampleActivity (rng: Random) (mu: float) (sigma: float) : float =
    LogNormal(mu, sigma, rng).Sample()

/// Sample trade size (stochastic rounding of activity sample)
let sampleSize (rng: Random) (activity: float) : int =
    max 1 (stochasticRound rng activity)

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
        let mu, sigma = activityMuSigma activityParams
        let correction = getVolatilityCorrection sigma
        let normal = Normal(0.0, 1.0, rng)
        let results = Array.zeroCreate timestamps.Length
        let mutable price = startPrice
        let mutable prevTime = 0.0
        
        for i in 0 .. timestamps.Length - 1 do
            let dt = timestamps.[i] - prevTime
            let rawSize = sampleActivity rng mu sigma
            let size = sampleSize rng rawSize
            // Normalize activity for volatility scaling (so E[sqrt(activity)] ~ 1)
            let activity = rawSize / activityParams.MeanSize
            
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
    
    let tradeCount = sampleTradeCount rng orderFlowParams durationSeconds
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
