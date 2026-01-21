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

/// Parameters for price generation within a trend (raw price changes, not %)
type TrendPriceParams = {
    DriftPerSecond: float           // Expected price change per second
    VolatilityPerSecond: float      // Std dev of price change per second
    IntraBarPointsMean: float       // Mean number of intra-bar price points
    IntraBarPointsStdDev: float     // Std dev of intra-bar points
}

/// Stochastic rounding: rounds up or down probabilistically based on fractional part
let stochasticRound (rng: Random) (x: float) : int =
    let floor = Math.Floor(x)
    let frac = x - floor
    if rng.NextDouble() < frac then int floor + 1 else int floor

/// Get price generation parameters for a trend type
let getTrendPriceParams (trend: Trend) : TrendPriceParams =
    match trend with
    | StrongUptrend ->   { DriftPerSecond = 0.003;  VolatilityPerSecond = 0.005; IntraBarPointsMean = 5.0; IntraBarPointsStdDev = 2.0 }
    | MidUptrend ->      { DriftPerSecond = 0.0015; VolatilityPerSecond = 0.004; IntraBarPointsMean = 4.0; IntraBarPointsStdDev = 2.0 }
    | WeakUptrend ->     { DriftPerSecond = 0.0007; VolatilityPerSecond = 0.003; IntraBarPointsMean = 3.0; IntraBarPointsStdDev = 1.5 }
    | Consolidation ->   { DriftPerSecond = 0.0;    VolatilityPerSecond = 0.002; IntraBarPointsMean = 2.0; IntraBarPointsStdDev = 1.0 }
    | WeakDowntrend ->   { DriftPerSecond = -0.0007; VolatilityPerSecond = 0.003; IntraBarPointsMean = 3.0; IntraBarPointsStdDev = 1.5 }
    | MidDowntrend ->    { DriftPerSecond = -0.0015; VolatilityPerSecond = 0.004; IntraBarPointsMean = 4.0; IntraBarPointsStdDev = 2.0 }
    | StrongDowntrend -> { DriftPerSecond = -0.003;  VolatilityPerSecond = 0.005; IntraBarPointsMean = 5.0; IntraBarPointsStdDev = 2.0 }

/// Generate price bars for a single trend episode
let generateTrendBars
    (rng: Random)
    (startTime: float)
    (startPrice: float)
    (session: DaySession)
    (trend: Trend)
    (durationMinutes: float)
    : Bar[] =
    
    let p = getTrendPriceParams trend
    let durationSeconds = int (durationMinutes * 60.0)
    let normal = Normal(0.0, 1.0, rng)
    let pointsDist = LogNormal.WithMeanVariance(p.IntraBarPointsMean, p.IntraBarPointsStdDev * p.IntraBarPointsStdDev, rng)
    
    let bars = Array.zeroCreate durationSeconds
    let mutable price = startPrice
    
    for i in 0 .. durationSeconds - 1 do
        let openPrice = price
        
        // Sample number of intra-bar price points
        let numPoints = stochasticRound rng (pointsDist.Sample())
        
        if numPoints = 0 then
            // No intra-bar movement
            bars.[i] <- {
                Time = startTime + float i
                Open = openPrice
                High = openPrice
                Low = openPrice
                Close = openPrice
                Session = session
                Trend = trend
            }
        else
            // Sample price points and track high/low
            let mutable high = openPrice
            let mutable low = openPrice
            
            for _ in 1 .. numPoints do
                let change = p.DriftPerSecond + normal.Sample() * p.VolatilityPerSecond
                price <- price + change
                high <- max high price
                low <- min low price
            
            bars.[i] <- {
                Time = startTime + float i
                Open = openPrice
                High = high
                Low = low
                Close = price
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
