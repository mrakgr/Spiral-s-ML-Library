#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

let rng = Random(42)

// =============================================================================
// Prototype: GBM with activity-correlated price-volume
//
// Current model (independent):
//   price_change ~ GBM with fixed volatility
//   size ~ Pareto (independent)
//
// New model (correlated):
//   activity ~ LogNormal(mu=-sigma²/2, sigma) with E[activity] = 1
//   size = base_size * activity
//   price_change ~ GBM but with volatility scaled by correction * sqrt(activity)
//
// The GBM formula becomes:
//   S(t+dt) = S(t) * exp((drift - vol²/2)*dt + vol*sqrt(dt)*Z)
//   
// With activity scaling on the volatility term:
//   S(t+dt) = S(t) * exp((drift - vol²/2)*dt + vol*correction*sqrt(activity)*sqrt(dt)*Z)
//
// Key insight: drift term stays the same (trend direction preserved),
// only the volatility term gets scaled by activity
// =============================================================================

type PriceParams = {
    DriftPerSecond: float
    VolatilityPerSecond: float
}

type ActivityParams = {
    Sigma: float           // LogNormal sigma (dispersion of activity)
    BaseSize: float        // Mean trade size
}

// Correction factor so E[correction * sqrt(activity)] = 1
let getCorrection (sigma: float) = exp(sigma * sigma / 8.0)

// Sample activity from LogNormal with E[activity] = 1
let sampleActivity (rng: Random) (sigma: float) =
    let mu = -sigma * sigma / 2.0
    LogNormal(mu, sigma, rng).Sample()

let stochasticRound (rng: Random) (x: float) : int =
    let floor = Math.Floor(x)
    let frac = x - floor
    int (if rng.NextDouble() < frac then floor + 1.0 else floor)

// Generate trades with correlated price-volume
let generateCorrelatedTrades 
    (rng: Random) 
    (priceParams: PriceParams) 
    (activityParams: ActivityParams)
    (startPrice: float) 
    (timestamps: float[]) 
    : (float * int)[] * float =
    
    if timestamps.Length = 0 then
        [||], startPrice
    else
        let correction = getCorrection activityParams.Sigma
        let normal = Normal(0.0, 1.0, rng)
        let results = Array.zeroCreate timestamps.Length
        let mutable price = startPrice
        let mutable prevTime = 0.0
        
        for i in 0 .. timestamps.Length - 1 do
            let dt = timestamps.[i] - prevTime
            let activity = sampleActivity rng activityParams.Sigma
            let size = max 1 (stochasticRound rng (activityParams.BaseSize * activity))
            
            let drift = priceParams.DriftPerSecond
            let baseVol = priceParams.VolatilityPerSecond
            let scaledVol = baseVol * correction * sqrt(activity)
            
            let z = normal.Sample()
            price <- price * exp((drift - baseVol * baseVol / 2.0) * dt + scaledVol * sqrt(dt) * z)
            results.[i] <- (price, size)
            prevTime <- timestamps.[i]
        
        results, price

// =============================================================================
// Test the correlated model
// =============================================================================

let generateTimestamps (rng: Random) (duration: float) (count: int) : float[] =
    let timestamps = Array.init count (fun _ -> rng.NextDouble() * duration)
    Array.sortInPlace timestamps
    timestamps

let priceParams = { DriftPerSecond = 30e-6; VolatilityPerSecond = 100e-6 }
let activityParams = { Sigma = 1.0; BaseSize = 100.0 }
let startPrice = 100.0
let durationSeconds = 60.0
let tradeCount = 3000

printfn "=== Correlated GBM Prototype ==="
printfn "Activity sigma: %.1f" activityParams.Sigma
printfn "Base size: %.0f" activityParams.BaseSize
printfn "Correction factor: %.4f" (getCorrection activityParams.Sigma)
printfn ""

let timestamps = generateTimestamps rng durationSeconds tradeCount
let results, endPrice = generateCorrelatedTrades rng priceParams activityParams startPrice timestamps

let prices = results |> Array.map fst
let sizes = results |> Array.map snd

printfn "=== Results ==="
printfn "Trades: %d" results.Length
printfn "Start price: %.4f, End price: %.4f" startPrice endPrice
printfn "Return: %.2f%%" ((endPrice - startPrice) / startPrice * 100.0)
printfn ""
printfn "Size stats:"
printfn "  Mean: %.1f (expected: %.1f)" (Array.averageBy float sizes) activityParams.BaseSize
printfn "  Max: %d" (Array.max sizes)
printfn ""

// Check correlation between size and |price change|
let priceChanges = Array.init (prices.Length - 1) (fun i -> abs (prices.[i+1] - prices.[i]))
let sizesForCorr = sizes.[1..] |> Array.map float

let meanSize = Array.average sizesForCorr
let meanChange = Array.average priceChanges
let cov = Array.map2 (fun s c -> (s - meanSize) * (c - meanChange)) sizesForCorr priceChanges |> Array.average
let stdSize = sqrt (Array.averageBy (fun s -> (s - meanSize) ** 2.0) sizesForCorr)
let stdChange = sqrt (Array.averageBy (fun c -> (c - meanChange) ** 2.0) priceChanges)
let corr = cov / (stdSize * stdChange)

printfn "Correlation(size, |price_change|): %.3f" corr

// =============================================================================
// Question: Should drift correction use baseVol or scaledVol?
//
// GBM: S(t+dt) = S(t) * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*Z)
// The -sigma²/2 term ensures E[S(t)] follows the drift exactly.
//
// If we scale sigma by activity, should we also scale the correction?
// Let's test both approaches.
// =============================================================================

printfn ""
printfn "=== Drift Correction Comparison ==="

// Run many episodes and compare average returns
let testDriftCorrection (useScaledCorrection: bool) =
    let mutable totalReturn = 0.0
    let runs = 500000
    let rng = Random(42)
    for _ in 1 .. runs do
        let ts = generateTimestamps rng durationSeconds 100
        let mutable price = startPrice
        let mutable prevTime = 0.0
        let correction = getCorrection activityParams.Sigma
        let normal = Normal(0.0, 1.0)
        
        for t in ts do
            let dt = t - prevTime
            let activity = sampleActivity rng activityParams.Sigma
            let scaledVol = priceParams.VolatilityPerSecond * correction * sqrt(activity)
            let driftCorr = if useScaledCorrection then scaledVol * scaledVol / 2.0 
                            else priceParams.VolatilityPerSecond * priceParams.VolatilityPerSecond / 2.0
            price <- price * exp((priceParams.DriftPerSecond - driftCorr) * dt + scaledVol * sqrt(dt) * normal.Sample())
            prevTime <- t
        totalReturn <- totalReturn + (price - startPrice) / startPrice
    totalReturn / float runs

let returnWithBase = testDriftCorrection false
let returnWithScaled = testDriftCorrection true
let expectedReturn = priceParams.DriftPerSecond * durationSeconds

printfn "Expected return (drift * duration): %.4f%%" (expectedReturn * 100.0)
printfn "Using baseVol² correction: %.4f%%" (returnWithBase * 100.0)
printfn "Using scaledVol² correction: %.4f%%" (returnWithScaled * 100.0)
