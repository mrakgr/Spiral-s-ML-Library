#r "nuget: MathNet.Numerics"

open System
open MathNet.Numerics.Distributions

type Trend = StrongUptrend | MidUptrend | WeakUptrend | Consolidation | WeakDowntrend | MidDowntrend | StrongDowntrend
type Episode = { Label: Trend; Duration: float }
type Trade = { Time: float; Trend: Trend }
type OrderFlowParams = { TradeRatePerSecond: float; DispersionExp: float }

let getOrderFlowParams (trend: Trend) : OrderFlowParams =
    match trend with
    | StrongUptrend ->   { TradeRatePerSecond = 50.0; DispersionExp = 1.0 }
    | MidUptrend ->      { TradeRatePerSecond = 40.0; DispersionExp = 1.0 }
    | WeakUptrend ->     { TradeRatePerSecond = 25.0; DispersionExp = 1.0 }
    | Consolidation ->   { TradeRatePerSecond = 10.0; DispersionExp = 1.0 }
    | WeakDowntrend ->   { TradeRatePerSecond = 25.0; DispersionExp = 1.0 }
    | MidDowntrend ->    { TradeRatePerSecond = 40.0; DispersionExp = 1.0 }
    | StrongDowntrend -> { TradeRatePerSecond = 50.0; DispersionExp = 1.0 }

let sampleTradeCount (rng: Random) (rate: float) (dispersionExp: float) (duration: float) =
    let p = Math.Pow(2.0, -dispersionExp)
    let r = rate * duration * p / (1.0 - p)
    let lambda = Gamma(r, (1.0 - p) / p, rng).Sample()
    Poisson(lambda, rng).Sample()

let generateTimestamps (rng: Random) (startTime: float) (duration: float) (count: int) : float[] =
    let timestamps = Array.init count (fun _ -> startTime + rng.NextDouble() * duration)
    Array.sortInPlace timestamps
    timestamps

let generateEpisodeTrades (rng: Random) (episode: Episode) : Trade[] =
    let durationSeconds = episode.Duration * 60.0
    let params' = getOrderFlowParams episode.Label
    let tradeCount = sampleTradeCount rng params'.TradeRatePerSecond params'.DispersionExp durationSeconds
    let timestamps = generateTimestamps rng 0.0 durationSeconds tradeCount
    timestamps |> Array.map (fun ts -> { Time = ts; Trend = episode.Label })

// Test
let rng = Random(42)

printfn "StrongUptrend (1 min):"
let trades1 = generateEpisodeTrades rng { Label = StrongUptrend; Duration = 1.0 }
printfn "  %d trades (expected ~3000)" trades1.Length

printfn "\nConsolidation (1 min):"
let trades2 = generateEpisodeTrades rng { Label = Consolidation; Duration = 1.0 }
printfn "  %d trades (expected ~600)" trades2.Length

printfn "\nFirst 5 trades from StrongUptrend:"
for t in trades1 |> Array.take 5 do
    printfn "  %.3f sec" t.Time
