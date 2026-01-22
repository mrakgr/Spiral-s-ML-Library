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

type TrendState = {
    Session: DaySession
    Trend: Trend
    mutable DurationRemaining: float  // in seconds
}

type TrendDistributions = {
    Params: TrendPriceParams
    Normal: Normal
    PointsDist: LogNormal
}

let createTrendDistributions (rng: Random) (mean: float) (trend: Trend) : TrendDistributions =
    let p = getTrendPriceParams trend
    {
        Params = p
        Normal = Normal(mean * p.DriftPerSecond, abs mean * p.VolatilityPerSecond, rng)
        PointsDist = LogNormal.WithMeanVariance(p.IntraBarPointsMean, p.IntraBarPointsStdDev * p.IntraBarPointsStdDev, rng)
    }

/// Generate price bars for a full day from DayResult
let generateDayBars (rng: Random) (startPrice: float) (result: DayResult) : Bar[] =
    let totalSeconds = 390 * 60
    let bars = Array.zeroCreate totalSeconds
    
    // Build queue of (session, trend, duration in seconds)
    let queue = Collections.Generic.Queue<TrendState>()
    for i in 0 .. result.Sessions.Length - 1 do
        let session = result.Sessions.[i]
        for trend in result.Trends.[i] do
            queue.Enqueue({ Session = session.Label; Trend = trend.Label; DurationRemaining = trend.Duration * 60.0 })
    
    let mutable mean = startPrice
    let mutable _open = startPrice
    let mutable current = queue.Dequeue()
    let mutable dist = createTrendDistributions rng mean current.Trend
    
    for i in 0 .. totalSeconds - 1 do
        // Sample bar
        let numPoints = stochasticRound rng (dist.PointsDist.Sample())
        mean <- mean + dist.Normal.Sample()
        
        let mutable close = _open
        let mutable high = _open
        let mutable low = _open
        
        for _ in 1 .. numPoints do
            close <- mean + dist.Normal.Sample()
            high <- max high close
            low <- min low close
        
        bars.[i] <- {
            Time = float i
            Open = _open
            High = high
            Low = low
            Close = close
            Session = current.Session
            Trend = current.Trend
        }
        _open <- close
        
        // Decrement duration and possibly switch to next trend
        current.DurationRemaining <- current.DurationRemaining - 1.0
        if current.DurationRemaining <= 0.0 && queue.Count > 0 then
            current <- queue.Dequeue()
            dist <- createTrendDistributions rng mean current.Trend
    
    bars

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
