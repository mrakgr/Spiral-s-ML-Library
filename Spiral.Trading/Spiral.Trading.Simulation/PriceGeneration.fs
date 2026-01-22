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
    | StrongUptrend ->   { DriftPerSecond = 30e-6 ;  VolatilityPerSecond = 100e-6; IntraBarPointsMean = 50.0; IntraBarPointsStdDev = 20.0 }
    | MidUptrend ->      { DriftPerSecond = 15e-6;   VolatilityPerSecond = 80e-6; IntraBarPointsMean = 40.0; IntraBarPointsStdDev = 20.0 }
    | WeakUptrend ->     { DriftPerSecond = 7e-6;    VolatilityPerSecond = 60e-6; IntraBarPointsMean = 30.0; IntraBarPointsStdDev = 15.0 }
    | Consolidation ->   { DriftPerSecond = 0.0;     VolatilityPerSecond = 40e-6; IntraBarPointsMean = 20.0; IntraBarPointsStdDev = 10.0 }
    | WeakDowntrend ->   { DriftPerSecond = -7e-6;   VolatilityPerSecond = 60e-6; IntraBarPointsMean = 30.0; IntraBarPointsStdDev = 15.0 }
    | MidDowntrend ->    { DriftPerSecond = -15e-6;  VolatilityPerSecond = 80e-6; IntraBarPointsMean = 40.0; IntraBarPointsStdDev = 20.0 }
    | StrongDowntrend -> { DriftPerSecond = -30e-6;  VolatilityPerSecond = 100e-6; IntraBarPointsMean = 50.0; IntraBarPointsStdDev = 20.0 }

type TrendBarResult = {
    TrendBars : Bar[]
    LatestMean : float
    DurationRemaining : float
}

/// Generate price bars for a single trend episode
let generateTrendBars
    (rng: Random)
    (startTime: float)
    (startMean: float)
    (lastClose : float)
    (session: DaySession)
    (trend: Trend)
    (durationMinutes: float)
    : TrendBarResult =
    
    let p = getTrendPriceParams trend
    let durationSeconds = durationMinutes * 60.0
    // Volatility should be relative to price.
    let normal = Normal(startMean * p.DriftPerSecond, abs startMean * p.VolatilityPerSecond, rng)
    let pointsDist = LogNormal.WithMeanVariance(p.IntraBarPointsMean, p.IntraBarPointsStdDev * p.IntraBarPointsStdDev, rng)
    
    let bars = Array.zeroCreate (int durationSeconds)
    let mutable volumeProfileMean = startMean
    let mutable _open = lastClose
    for i in 0 .. int durationSeconds - 1 do
        // Sample number of intra-bar price points
        let numPoints = stochasticRound rng (pointsDist.Sample())
        // Sample a new price bar volume profile mean.
        volumeProfileMean <- volumeProfileMean + normal.Sample()
        
        // Sample price points and track high/low
        let mutable close = _open
        let mutable high = _open
        let mutable low = _open
        
        for _ in 1 .. numPoints do
            close <- volumeProfileMean + normal.Sample()
            high <- max high close
            low <- min low close
        
        bars.[i] <- {
            Time = startTime + float i
            Open = _open
            High = high
            Low = low
            Close = close
            Session = session
            Trend = trend
        }
        _open <- close
    
    {
        TrendBars = bars
        LatestMean = volumeProfileMean
        DurationRemaining = durationSeconds - float (int durationSeconds)
    }

/// Generate price bars for a full day from DayResult
let generateDayBars (rng: Random) (startPrice: float) (result: DayResult) : Bar[] =
    let allBars = ResizeArray<Bar>()
    let mutable time = 0.0
    let mutable mean = startPrice
    let mutable lastClose = startPrice
    let mutable durationRemainingSeconds = 0.0
    
    for i in 0 .. result.Sessions.Length - 1 do
        let session = result.Sessions.[i]
        let trends = result.Trends.[i]
        for trend in trends do
            let totalSeconds = trend.Duration * 60.0 + durationRemainingSeconds
            let r = generateTrendBars rng time mean lastClose session.Label trend.Label (totalSeconds / 60.0)
            if r.TrendBars.Length > 0 then
                allBars.AddRange(r.TrendBars)
                time <- r.TrendBars.[r.TrendBars.Length - 1].Time + 1.0
                lastClose <- r.TrendBars.[r.TrendBars.Length - 1].Close
            mean <- r.LatestMean
            durationRemainingSeconds <- r.DurationRemaining
    
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
