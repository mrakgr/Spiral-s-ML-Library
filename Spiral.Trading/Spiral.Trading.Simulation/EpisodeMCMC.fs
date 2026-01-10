module Spiral.Trading.Simulation.EpisodeMCMC

open System
open MathNet.Numerics.Distributions
open Spiral.Trading.Simulation.Episode

type SessionState = (DaySession * int)[]

type SessionParams = {
    Mean: float
    StdDev: float
}

type MCMCConfig = {
    MorningParams: SessionParams
    MidParams: SessionParams
    CloseParams: SessionParams
    MinSessionLength: int
    MaxDelta: int
    Iterations: int
}

let defaultMCMCConfig = {
    MorningParams = { Mean = 60.0; StdDev = 20.0 }
    MidParams = { Mean = 270.0; StdDev = 40.0 }
    CloseParams = { Mean = 60.0; StdDev = 20.0 }
    MinSessionLength = 1
    MaxDelta = 10
    Iterations = 10000
}

/// Convert mean/stdDev to log-normal mu/sigma parameters
let logNormalParams (mean: float) (stdDev: float) : float * float =
    let variance = stdDev * stdDev
    let sigma2 = log(1.0 + variance / (mean * mean))
    let sigma = sqrt(sigma2)
    let mu = log(mean) - sigma2 / 2.0
    (mu, sigma)

/// Compute log-likelihood of a duration under log-normal distribution
let logLikelihoodSession (sessionParams: SessionParams) (duration: int) : float =
    let (mu, sigma) = logNormalParams sessionParams.Mean sessionParams.StdDev
    let dist = LogNormal(mu, sigma)
    dist.DensityLn(float duration)

/// Compute total log-likelihood of a session state
let logLikelihood (config: MCMCConfig) (state: SessionState) : float =
    state
    |> Array.sumBy (fun (session, duration) ->
        let sp =
            match session with
            | Morning -> config.MorningParams
            | Mid -> config.MidParams
            | Close -> config.CloseParams
        logLikelihoodSession sp duration)

/// Propose a new state by moving duration between adjacent sessions
let proposeMove (config: MCMCConfig) (state: SessionState) (rng: Random) : SessionState option =
    // Pick which boundary to adjust (0 = Morning/Mid, 1 = Mid/Close)
    let boundaryIdx = rng.Next(2)
    let idx1 = boundaryIdx
    let idx2 = boundaryIdx + 1
    
    // Pick random delta from -MaxDelta to +MaxDelta (excluding 0)
    let delta = 
        let d = rng.Next(1, config.MaxDelta + 1)
        if rng.NextDouble() < 0.5 then -d else d
    
    let (session1, dur1) = state.[idx1]
    let (session2, dur2) = state.[idx2]
    
    let newDur1 = dur1 + delta
    let newDur2 = dur2 - delta
    
    // Check validity
    if newDur1 >= config.MinSessionLength && newDur2 >= config.MinSessionLength then
        let newState = Array.copy state
        newState.[idx1] <- (session1, newDur1)
        newState.[idx2] <- (session2, newDur2)
        Some newState
    else
        None

/// Run Metropolis-Hastings MCMC
let runMCMC (config: MCMCConfig) (initial: SessionState) (rng: Random) : SessionState =
    let mutable current = initial
    let mutable currentLL = logLikelihood config current
    
    for _ in 1 .. config.Iterations do
        match proposeMove config current rng with
        | Some proposed ->
            let proposedLL = logLikelihood config proposed
            let logAcceptRatio = proposedLL - currentLL
            
            if log(rng.NextDouble()) < logAcceptRatio then
                current <- proposed
                currentLL <- proposedLL
        | None ->
            () // Invalid move, reject
    
    current

/// Create the initial state
let initialState () : SessionState =
    [| (Morning, 60); (Mid, 270); (Close, 60) |]

/// Print session state
let printState (state: SessionState) : unit =
    printfn "Session State (total %d min):" (state |> Array.sumBy snd)
    let mutable t = 0
    for (session, duration) in state do
        let name = 
            match session with
            | Morning -> "Morning"
            | Mid -> "Mid"
            | Close -> "Close"
        printfn "  %3d-%3d: %-8s (%d min)" t (t + duration - 1) name duration
        t <- t + duration
