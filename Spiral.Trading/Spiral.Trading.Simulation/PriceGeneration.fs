module Spiral.Trading.Simulation.PriceGeneration

open System
open MathNet.Numerics.Distributions
open Spiral.Trading.Simulation.EpisodeMCMC

/// A single price bar (1-second resolution)
type Bar = {
    Time: float      // Seconds from start
    Open: float
    High: float
    Low: float
    Close: float
    Session: DaySession
    Trend: Trend
}

/// Parameters for price generation within a trend
type TrendPriceParams = {
    DriftPerMinute: float      // Expected return per minute (e.g., 0.001 = 0.1%)
    VolatilityPerSecond: float // Standard deviation per second (e.g., 0.0005 = 0.05%)
}

/// Get price generation parameters for a trend type
let getTrendPriceParams (trend: Trend) : TrendPriceParams =
    match trend with
    | StrongUptrend ->   { DriftPerMinute = 0.002;  VolatilityPerSecond = 0.0006 }
    | MidUptrend ->      { DriftPerMinute = 0.001;  VolatilityPerSecond = 0.0005 }
    | WeakUptrend ->     { DriftPerMinute = 0.0005; VolatilityPerSecond = 0.0004 }
    | Consolidation ->   { DriftPerMinute = 0.0;    VolatilityPerSecond = 0.0003 }
    | WeakDowntrend ->   { DriftPerMinute = -0.0005; VolatilityPerSecond = 0.0004 }
    | MidDowntrend ->    { DriftPerMinute = -0.001;  VolatilityPerSecond = 0.0005 }
    | StrongDowntrend -> { DriftPerMinute = -0.002;  VolatilityPerSecond = 0.0006 }

/// Generate price bars for a single trend episode
let generateTrendBars
    (rng: Random)
    (startTime: float)
    (startPrice: float)
    (session: DaySession)
    (trend: Trend)
    (durationMinutes: float)
    : Bar[] =
    
    let priceParams = getTrendPriceParams trend
    let durationSeconds = int (durationMinutes * 60.0)
    let driftPerSecond = priceParams.DriftPerMinute / 60.0
    let normal = Normal(0.0, 1.0, rng)
    
    let bars = Array.zeroCreate durationSeconds
    let mutable price = startPrice
    
    for i in 0 .. durationSeconds - 1 do
        let openPrice = price
        
        let noise = normal.Sample() * priceParams.VolatilityPerSecond
        let returnRate = driftPerSecond + noise
        price <- price * (1.0 + returnRate)
        
        let closePrice = price
        let high = max openPrice closePrice * (1.0 + abs(normal.Sample()) * priceParams.VolatilityPerSecond * 0.5)
        let low = min openPrice closePrice * (1.0 - abs(normal.Sample()) * priceParams.VolatilityPerSecond * 0.5)
        
        bars.[i] <- {
            Time = startTime + float i
            Open = openPrice
            High = high
            Low = low
            Close = closePrice
            Session = session
            Trend = trend
        }
    
    bars

/// Generate price bars for a full day from DayResult
let generateDayBars (rng: Random) (startPrice: float) (result: DayResult) : Bar[] =
    let allBars = ResizeArray<Bar>()
    let mutable time = 0.0
    let mutable price = startPrice
    
    for i in 0 .. result.Sessions.Length - 1 do
        let session = result.Sessions.[i]
        let trends = result.Trends.[i]
        for trend in trends do
            let bars = generateTrendBars rng time price session.Label trend.Label trend.Duration
            if bars.Length > 0 then
                allBars.AddRange(bars)
                time <- bars.[bars.Length - 1].Time + 1.0
                price <- bars.[bars.Length - 1].Close
    
    allBars.ToArray()

/// Print summary statistics for generated bars
let printBarsSummary (bars: Bar[]) : unit =
    if bars.Length = 0 then
        printfn "No bars generated"
    else
        let first = bars.[0]
        let last = bars.[bars.Length - 1]
        let high = bars |> Array.map (fun b -> b.High) |> Array.max
        let low = bars |> Array.map (fun b -> b.Low) |> Array.min
        let returnPct = (last.Close - first.Open) / first.Open * 100.0
        
        printfn "Price Summary:"
        printfn "  Bars: %d (%.1f minutes)" bars.Length (float bars.Length / 60.0)
        printfn "  Open:  %.4f" first.Open
        printfn "  Close: %.4f" last.Close
        printfn "  High:  %.4f" high
        printfn "  Low:   %.4f" low
        printfn "  Return: %.2f%%" returnPct

/// Export bars to CSV file
let exportToCsv (path: string) (bars: Bar[]) : unit =
    use writer = new System.IO.StreamWriter(path)
    writer.WriteLine("Time,Open,High,Low,Close,Session,Trend")
    for bar in bars do
        writer.WriteLine(sprintf "%.0f,%.6f,%.6f,%.6f,%.6f,%A,%A" 
            bar.Time bar.Open bar.High bar.Low bar.Close bar.Session bar.Trend)
    printfn "Exported %d bars to %s" bars.Length path
