module Spiral.Trading.Simulation.Episode

open System
open MathNet.Numerics.Distributions

type DaySession = 
    | Morning
    | Mid
    | Close

type Trend =
    | StrongUptrend
    | MidUptrend
    | WeakUptrend
    | Consolidation
    | WeakDowntrend
    | MidDowntrend
    | StrongDowntrend

type DaySessionParams = {
    DayLengthMinutes: int
    MorningEndMean: float
    MorningEndStdDev: float
    CloseStartMean: float
    CloseStartStdDev: float
}

type TrendParams = {
    DurationMean: float
    DurationStdDev: float
}

module Hazard =
    /// Compute stopping probability at time t given a CDF
    /// P(stop at t | survived to t) = (CDF(t+1) - CDF(t)) / (1 - CDF(t))
    let stoppingProbability (cdf: float -> float) (t: int) : float =
        let F_t = cdf (float t)
        let F_t1 = cdf (float (t + 1))
        let survival = 1.0 - F_t
        if survival > 1e-10 then
            (F_t1 - F_t) / survival
        else
            1.0

    /// Build stopping probabilities array from any CDF
    let buildStoppingProbabilities (cdf: float -> float) (maxSteps: int) : float[] =
        [| for t in 0 .. maxSteps - 1 -> stoppingProbability cdf t |]

    /// Build CDF for normal distribution
    let normalCdf (mean: float) (stdDev: float) : float -> float =
        let dist = Normal(mean, stdDev)
        dist.CumulativeDistribution

    /// Build CDF for log-normal distribution
    /// Parameters are mean and stdDev of the underlying normal (mu, sigma)
    let logNormalCdf (mu: float) (sigma: float) : float -> float =
        let dist = LogNormal(mu, sigma)
        dist.CumulativeDistribution

    /// Convert desired mean/stdDev to log-normal mu/sigma parameters
    let logNormalParams (mean: float) (stdDev: float) : float * float =
        let variance = stdDev * stdDev
        let mu = log(mean * mean / sqrt(variance + mean * mean))
        let sigma = sqrt(log(1.0 + variance / (mean * mean)))
        (mu, sigma)

/// Selection probabilities for trends based on day session
let getTrendSelectionWeights (session: DaySession) : Map<Trend, float> =
    match session with
    | Morning | Close ->
        Map.ofList [
            StrongUptrend, 0.15
            MidUptrend, 0.15
            WeakUptrend, 0.10
            Consolidation, 0.20
            WeakDowntrend, 0.10
            MidDowntrend, 0.15
            StrongDowntrend, 0.15
        ]
    | Mid ->
        Map.ofList [
            StrongUptrend, 0.02
            MidUptrend, 0.08
            WeakUptrend, 0.15
            Consolidation, 0.50
            WeakDowntrend, 0.15
            MidDowntrend, 0.08
            StrongDowntrend, 0.02
        ]

/// Duration parameters for each trend type
let getTrendDurationParams (trend: Trend) : TrendParams =
    match trend with
    | StrongUptrend | StrongDowntrend -> { DurationMean = 5.0; DurationStdDev = 2.0 }
    | MidUptrend | MidDowntrend -> { DurationMean = 15.0; DurationStdDev = 5.0 }
    | WeakUptrend | WeakDowntrend -> { DurationMean = 30.0; DurationStdDev = 10.0 }
    | Consolidation -> { DurationMean = 20.0; DurationStdDev = 10.0 }

/// Sample a trend based on selection weights using MathNet Categorical distribution
let sampleTrend (weights: Map<Trend, float>) (rng: Random) : Trend =
    let allTrends, probs = weights |> Map.toArray |> Array.unzip
    let dist = Categorical(probs, rng)
    allTrends.[dist.Sample()]

/// Simulate a full trading day, returning the session state at each minute
let simulateDay (config: DaySessionParams) (rng: Random) : DaySession[] =
    let morningCdf = Hazard.normalCdf config.MorningEndMean config.MorningEndStdDev
    let closeCdf = Hazard.normalCdf config.CloseStartMean config.CloseStartStdDev
    
    let result = Array.zeroCreate config.DayLengthMinutes
    let mutable currentState = Morning
    
    for t in 0 .. config.DayLengthMinutes - 1 do
        result.[t] <- currentState
        
        match currentState with
        | Morning ->
            let stopProb = Hazard.stoppingProbability morningCdf t
            if rng.NextDouble() < stopProb then
                currentState <- Mid
        | Mid ->
            let stopProb = Hazard.stoppingProbability closeCdf t
            if rng.NextDouble() < stopProb then
                currentState <- Close
        | Close ->
            () // Stay in close until end of day
    
    result

/// Simulate subepisodes (trends) for a day given the session structure
let simulateTrends (sessions: DaySession[]) (rng: Random) : Trend[] =
    let result = Array.zeroCreate sessions.Length
    let mutable currentTrend = Consolidation
    let mutable trendElapsed = 0
    let mutable trendCdf = Hazard.logNormalCdf 3.0 0.5  // default
    
    for t in 0 .. sessions.Length - 1 do
        let session = sessions.[t]
        
        // Check if current trend should end
        let stopProb = Hazard.stoppingProbability trendCdf trendElapsed
        let shouldStop = rng.NextDouble() < stopProb
        
        if shouldStop || t = 0 then
            // Select new trend based on current session
            let weights = getTrendSelectionWeights session
            currentTrend <- sampleTrend weights rng
            let durationParams = getTrendDurationParams currentTrend
            let mu, sigma = Hazard.logNormalParams durationParams.DurationMean durationParams.DurationStdDev
            trendCdf <- Hazard.logNormalCdf mu sigma
            trendElapsed <- 0
        
        result.[t] <- currentTrend
        trendElapsed <- trendElapsed + 1
    
    result

/// Default parameters for a typical trading day
let defaultDayParams : DaySessionParams = {
    DayLengthMinutes = 390  // 6.5 hours
    MorningEndMean = 60.0   // 1 hour
    MorningEndStdDev = 20.0
    CloseStartMean = 330.0  // 1 hour before close
    CloseStartStdDev = 20.0
}

/// Group consecutive elements and return (element, count) pairs
let groupConsecutive (arr: 'a[]) : ('a * int) list =
    if arr.Length = 0 then []
    else
        let mutable result = []
        let mutable current = arr.[0]
        let mutable count = 1
        for i in 1 .. arr.Length - 1 do
            if arr.[i] = current then
                count <- count + 1
            else
                result <- (current, count) :: result
                current <- arr.[i]
                count <- 1
        result <- (current, count) :: result
        List.rev result

/// Print a summary of the day simulation
let printDaySummary (sessions: DaySession[]) : unit =
    let morningEnd = 
        sessions 
        |> Array.tryFindIndex (fun s -> s <> Morning)
        |> Option.defaultValue sessions.Length
    
    let closeStart = 
        sessions 
        |> Array.tryFindIndex (fun s -> s = Close)
        |> Option.defaultValue sessions.Length
    
    printfn "Day Session Summary:"
    printfn "  Morning: 0 - %d minutes (%d min)" (morningEnd - 1) morningEnd
    printfn "  Mid:     %d - %d minutes (%d min)" morningEnd (closeStart - 1) (closeStart - morningEnd)
    printfn "  Close:   %d - %d minutes (%d min)" closeStart (sessions.Length - 1) (sessions.Length - closeStart)

/// Print trend summary with consecutive grouping
let printTrendSummary (trends: Trend[]) : unit =
    let groups = groupConsecutive trends
    printfn "Trend Summary (%d episodes):" groups.Length
    let mutable startTime = 0
    for (trend, count) in groups do
        let trendName = 
            match trend with
            | StrongUptrend -> "StrongUp"
            | MidUptrend -> "MidUp"
            | WeakUptrend -> "WeakUp"
            | Consolidation -> "Consol"
            | WeakDowntrend -> "WeakDown"
            | MidDowntrend -> "MidDown"
            | StrongDowntrend -> "StrongDown"
        printfn "  %3d-%3d: %-10s (%d min)" startTime (startTime + count - 1) trendName count
        startTime <- startTime + count
