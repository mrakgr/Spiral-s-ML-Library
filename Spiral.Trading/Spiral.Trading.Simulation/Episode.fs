module Spiral.Trading.Simulation.Episode

open System
open MathNet.Numerics.Distributions

type DaySession = 
    | Morning
    | Mid
    | Close

type DaySessionParams = {
    DayLengthMinutes: int
    MorningEndMean: float
    MorningEndStdDev: float
    CloseStartMean: float
    CloseStartStdDev: float
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

/// Default parameters for a typical trading day
let defaultDayParams : DaySessionParams = {
    DayLengthMinutes = 390  // 6.5 hours
    MorningEndMean = 60.0   // 1 hour
    MorningEndStdDev = 20.0
    CloseStartMean = 330.0  // 1 hour before close
    CloseStartStdDev = 20.0
}

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
    
    printfn "Day Simulation Summary:"
    printfn "  Morning: 0 - %d minutes" (morningEnd - 1)
    printfn "  Mid:     %d - %d minutes" morningEnd (closeStart - 1)
    printfn "  Close:   %d - %d minutes" closeStart (sessions.Length - 1)
