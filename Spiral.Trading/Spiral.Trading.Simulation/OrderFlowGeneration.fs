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

/// Parameters for price generation (time-normalized, then applied per-trade)
type PriceParams = {
    DriftPerSecond: float        // Drift per second (normalized to mean trade rate and size)
    VolatilityPerSecond: float   // Volatility per second (normalized to mean trade rate and size)
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

/// Get price parameters for a trend type (drift and volatility per second)
/// These match the original time-based model values
let getPriceParams (trend: Trend) : PriceParams =
    match trend with
    | StrongUptrend ->   { DriftPerSecond = 30e-6;  VolatilityPerSecond = 300e-6 }
    | MidUptrend ->      { DriftPerSecond = 15e-6;  VolatilityPerSecond = 240e-6 }
    | WeakUptrend ->     { DriftPerSecond = 7e-6;   VolatilityPerSecond = 180e-6 }
    | Consolidation ->   { DriftPerSecond = 0.0;    VolatilityPerSecond = 120e-6 }
    | WeakDowntrend ->   { DriftPerSecond = -7e-6;  VolatilityPerSecond = 180e-6 }
    | MidDowntrend ->    { DriftPerSecond = -15e-6; VolatilityPerSecond = 240e-6 }
    | StrongDowntrend -> { DriftPerSecond = -30e-6; VolatilityPerSecond = 300e-6 }

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

/// Sample size from LogNormal distribution
let sampleSize (rng: Random) (mu: float) (sigma: float) : int =
    let rec loop() =
        let rawSize = LogNormal(mu, sigma, rng).Sample()
        let size = rawSize |> stochasticRound rng
        if size > 0 then size else loop()
    loop()

/// Generate uniformly distributed timestamps within an interval
let generateTimestamps (rng: Random) (startTime: float) (duration: float) (count: int) : float[] =
    let timestamps = Array.init count (fun _ -> startTime + rng.NextDouble() * duration)
    Array.sortInPlace timestamps
    timestamps

/// Correction factor so E[correction * sqrt(size/meanSize)] = 1 for LogNormal
let getActivityCorrection (sigma: float) : float =
    exp(sigma * sigma / 8.0)

/// Generate prices and sizes using volume-based GBM with activity scaling
/// Returns array of (price, size) pairs and the final price for chaining
let generatePricesAndSizes 
    (rng: Random) 
    (priceParams: PriceParams) 
    (orderFlowParams: OrderFlowParams)
    (activityParams: ActivityParams)
    (startPrice: float) 
    (count: int) 
    : (float * int)[] * float =
    
    if count = 0 then
        [||], startPrice
    else
        let mu, sigma = activityMuSigma activityParams
        let correction = getActivityCorrection sigma
        let normal = Normal(0.0, 1.0, rng)
        let results = Array.zeroCreate count
        let mutable logPrice = log(startPrice)
        
        let expectedDt = 1.0 / orderFlowParams.MeanTradesPerSecond
        let sqrtExpectedDt = sqrt(expectedDt)
        let expectedSqrtSize = sqrt(activityParams.MeanSize)
        let drift = priceParams.DriftPerSecond
        let vol = priceParams.VolatilityPerSecond
        
        for i in 0 .. count - 1 do
            let size = sampleSize rng mu sigma
            let sizeNorm = correction * sqrt(float size) / expectedSqrtSize
            let scaledDrift = drift * sizeNorm
            let scaledVol = vol * sizeNorm
            let z = normal.Sample()
            logPrice <- logPrice + (scaledDrift - scaledVol * scaledVol / 2.0) * expectedDt + scaledVol * sqrtExpectedDt * z
            results.[i] <- (exp(logPrice), size)
        
        results, exp(logPrice)

/// Generate trades for a single trend episode
/// Returns trades and the ending price for chaining to next episode
let generateEpisodeTrades (rng: Random) (startPrice: float) (episode: Episode<Trend>) : Trade[] * float =
    let durationSeconds = episode.Duration * 60.0
    let orderFlowParams = getOrderFlowParams episode.Label
    let priceParams = getPriceParams episode.Label
    let activityParams = getActivityParams episode.Label
    
    let tradeCount = sampleTradeCount rng orderFlowParams durationSeconds
    let timestamps = generateTimestamps rng 0.0 durationSeconds tradeCount
    let pricesAndSizes, endPrice = generatePricesAndSizes rng priceParams orderFlowParams activityParams startPrice tradeCount
    
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
