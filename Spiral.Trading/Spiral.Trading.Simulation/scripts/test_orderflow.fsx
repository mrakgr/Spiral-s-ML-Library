#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

type Trend = StrongUptrend | MidUptrend | WeakUptrend | Consolidation | WeakDowntrend | MidDowntrend | StrongDowntrend
type Episode = { Label: Trend; Duration: float }
type Trade = { Time: float; Price: float; Size: int; Trend: Trend }
type OrderFlowParams = { TradeRatePerSecond: float; DispersionExp: float }
type PriceParams = { DriftPerSecond: float; VolatilityPerSecond: float }
type SizeParams = { MinSize: float; Alpha: float }

let getOrderFlowParams (trend: Trend) : OrderFlowParams =
    match trend with
    | StrongUptrend ->   { TradeRatePerSecond = 50.0; DispersionExp = 1.0 }
    | MidUptrend ->      { TradeRatePerSecond = 40.0; DispersionExp = 1.0 }
    | WeakUptrend ->     { TradeRatePerSecond = 25.0; DispersionExp = 1.0 }
    | Consolidation ->   { TradeRatePerSecond = 10.0; DispersionExp = 1.0 }
    | WeakDowntrend ->   { TradeRatePerSecond = 25.0; DispersionExp = 1.0 }
    | MidDowntrend ->    { TradeRatePerSecond = 40.0; DispersionExp = 1.0 }
    | StrongDowntrend -> { TradeRatePerSecond = 50.0; DispersionExp = 1.0 }

let getPriceParams (trend: Trend) : PriceParams =
    match trend with
    | StrongUptrend ->   { DriftPerSecond = 30e-6;  VolatilityPerSecond = 100e-6 }
    | MidUptrend ->      { DriftPerSecond = 15e-6;  VolatilityPerSecond = 80e-6 }
    | WeakUptrend ->     { DriftPerSecond = 7e-6;   VolatilityPerSecond = 60e-6 }
    | Consolidation ->   { DriftPerSecond = 0.0;    VolatilityPerSecond = 40e-6 }
    | WeakDowntrend ->   { DriftPerSecond = -7e-6;  VolatilityPerSecond = 60e-6 }
    | MidDowntrend ->    { DriftPerSecond = -15e-6; VolatilityPerSecond = 80e-6 }
    | StrongDowntrend -> { DriftPerSecond = -30e-6; VolatilityPerSecond = 100e-6 }

let getSizeParams (trend: Trend) : SizeParams =
    match trend with
    | StrongUptrend ->   { MinSize = 1.0; Alpha = 1.5 }
    | MidUptrend ->      { MinSize = 1.0; Alpha = 1.8 }
    | WeakUptrend ->     { MinSize = 1.0; Alpha = 2.0 }
    | Consolidation ->   { MinSize = 1.0; Alpha = 2.5 }
    | WeakDowntrend ->   { MinSize = 1.0; Alpha = 2.0 }
    | MidDowntrend ->    { MinSize = 1.0; Alpha = 1.8 }
    | StrongDowntrend -> { MinSize = 1.0; Alpha = 1.5 }

let sampleTradeCount (rng: Random) (rate: float) (dispersionExp: float) (duration: float) =
    let p = Math.Pow(2.0, -dispersionExp)
    let r = rate * duration * p / (1.0 - p)
    let lambda = Gamma(r, (1.0 - p) / p, rng).Sample()
    Poisson(lambda, rng).Sample()

let sampleSize (rng: Random) (sizeParams: SizeParams) : int =
    let pareto = Pareto(sizeParams.MinSize, sizeParams.Alpha, rng)
    max 1 (int (pareto.Sample()))

let generateTimestamps (rng: Random) (startTime: float) (duration: float) (count: int) : float[] =
    let timestamps = Array.init count (fun _ -> startTime + rng.NextDouble() * duration)
    Array.sortInPlace timestamps
    timestamps

let generatePrices (rng: Random) (priceParams: PriceParams) (startPrice: float) (timestamps: float[]) : float[] * float =
    if timestamps.Length = 0 then [||], startPrice
    else
        let prices = Array.zeroCreate timestamps.Length
        let normal = Normal(0.0, 1.0, rng)
        let mutable price = startPrice
        let mutable prevTime = 0.0
        for i in 0 .. timestamps.Length - 1 do
            let dt = timestamps.[i] - prevTime
            let drift = priceParams.DriftPerSecond
            let vol = priceParams.VolatilityPerSecond
            let z = normal.Sample()
            price <- price * exp((drift - vol * vol / 2.0) * dt + vol * sqrt(dt) * z)
            prices.[i] <- price
            prevTime <- timestamps.[i]
        prices, price

let generateEpisodeTrades (rng: Random) (startPrice: float) (episode: Episode) : Trade[] * float =
    let durationSeconds = episode.Duration * 60.0
    let orderFlowParams = getOrderFlowParams episode.Label
    let priceParams = getPriceParams episode.Label
    let sizeParams = getSizeParams episode.Label
    let tradeCount = sampleTradeCount rng orderFlowParams.TradeRatePerSecond orderFlowParams.DispersionExp durationSeconds
    let timestamps = generateTimestamps rng 0.0 durationSeconds tradeCount
    let prices, endPrice = generatePrices rng priceParams startPrice timestamps
    let trades = Array.init tradeCount (fun i -> {
        Time = timestamps.[i]; Price = prices.[i]; Size = sampleSize rng sizeParams; Trend = episode.Label
    })
    trades, endPrice

// Test
let rng = Random(42)
let startPrice = 100.0

printfn "=== StrongUptrend (1 min) ==="
let trades1, endPrice1 = generateEpisodeTrades rng startPrice { Label = StrongUptrend; Duration = 1.0 }
printfn "Trades: %d (expected ~3000)" trades1.Length
printfn "Start: %.4f, End: %.4f, Return: %.2f%%" startPrice endPrice1 ((endPrice1 - startPrice) / startPrice * 100.0)
let totalVol1 = trades1 |> Array.sumBy (fun t -> t.Size)
let avgSize1 = float totalVol1 / float trades1.Length
let maxSize1 = trades1 |> Array.map (fun t -> t.Size) |> Array.max
printfn "Volume: %d, Avg size: %.1f, Max: %d" totalVol1 avgSize1 maxSize1

printfn "\n=== Consolidation (1 min) ==="
let trades2, endPrice2 = generateEpisodeTrades rng startPrice { Label = Consolidation; Duration = 1.0 }
printfn "Trades: %d (expected ~600)" trades2.Length
printfn "Start: %.4f, End: %.4f, Return: %.2f%%" startPrice endPrice2 ((endPrice2 - startPrice) / startPrice * 100.0)
let totalVol2 = trades2 |> Array.sumBy (fun t -> t.Size)
let avgSize2 = float totalVol2 / float trades2.Length
let maxSize2 = trades2 |> Array.map (fun t -> t.Size) |> Array.max
printfn "Volume: %d, Avg size: %.1f, Max: %d" totalVol2 avgSize2 maxSize2

printfn "\n=== First 10 trades from StrongUptrend ==="
for t in trades1 |> Array.take 10 do
    printfn "  t=%.3fs  price=%.4f  size=%d" t.Time t.Price t.Size
