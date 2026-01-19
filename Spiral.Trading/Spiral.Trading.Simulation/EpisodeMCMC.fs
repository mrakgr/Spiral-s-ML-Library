module Spiral.Trading.Simulation.EpisodeMCMC

open System
open MathNet.Numerics.Distributions

// =============================================================================
// Core Types
// =============================================================================

/// Generic episode with a label and duration
type Episode<'label> = {
    Label: 'label
    Duration: float
}

/// Day session types
type DaySession =
    | Morning
    | Mid
    | Close

/// Trend types within sessions
type Trend =
    | StrongUptrend
    | MidUptrend
    | WeakUptrend
    | Consolidation
    | WeakDowntrend
    | MidDowntrend
    | StrongDowntrend

// =============================================================================
// Generic MCMC Module
// =============================================================================

module MCMC =
    type Config = {
        Iterations: int
    }

    let defaultConfig = { Iterations = 10000 }

    /// Run Metropolis-Hastings MCMC sampler
    /// Returns a single sample from the posterior after running for the specified iterations
    let run
        (config: Config)
        (logLikelihood: 'state -> float)
        (propose: Random -> 'state -> 'state option)
        (initial: 'state)
        (rng: Random)
        : 'state =

        let mutable current = initial
        let mutable currentLL = logLikelihood current

        for _ in 1 .. config.Iterations do
            match propose rng current with
            | Some proposed ->
                let proposedLL = logLikelihood proposed
                let logAcceptRatio = proposedLL - currentLL

                if log(rng.NextDouble()) < logAcceptRatio then
                    current <- proposed
                    currentLL <- proposedLL
            | None ->
                () // Invalid move, reject

        current

// =============================================================================
// Distribution Utilities
// =============================================================================

module Distribution =
    /// Convert mean/stdDev to log-normal mu/sigma parameters
    let logNormalParams (mean: float) (stdDev: float) : float * float =
        let variance = stdDev * stdDev
        let sigma2 = log(1.0 + variance / (mean * mean))
        let sigma = sqrt(sigma2)
        let mu = log(mean) - sigma2 / 2.0
        (mu, sigma)

    /// Compute log-likelihood under log-normal distribution
    let logNormalLogLikelihood (mean: float) (stdDev: float) (value: float) : float =
        if value <= 0.0 then
            Double.NegativeInfinity
        else
            let (mu, sigma) = logNormalParams mean stdDev
            let dist = LogNormal(mu, sigma)
            dist.DensityLn(value)

// =============================================================================
// Session Level (Day -> Sessions)
// =============================================================================

module SessionLevel =
    type Params = {
        Mean: float
        StdDev: float
    }

    type Config = {
        MorningParams: Params
        MidParams: Params
        CloseParams: Params
        MinDuration: float
        MaxDelta: float
    }

    let defaultConfig = {
        MorningParams = { Mean = 60.0; StdDev = 20.0 }
        MidParams = { Mean = 270.0; StdDev = 40.0 }
        CloseParams = { Mean = 60.0; StdDev = 20.0 }
        MinDuration = 1.0
        MaxDelta = 10.0
    }

    /// State for session-level MCMC: array of (session, duration) pairs
    type State = Episode<DaySession>[]

    let private getParams (config: Config) (session: DaySession) : Params =
        match session with
        | Morning -> config.MorningParams
        | Mid -> config.MidParams
        | Close -> config.CloseParams

    /// Compute log-likelihood for a session state
    let logLikelihood (config: Config) (state: State) : float =
        state
        |> Array.sumBy (fun ep ->
            let p = getParams config ep.Label
            Distribution.logNormalLogLikelihood p.Mean p.StdDev ep.Duration)

    /// Propose a move by transferring duration between adjacent sessions
    let propose (config: Config) (rng: Random) (state: State) : State option =
        // Pick two sessions to transfer duration between
        let idx1 = rng.Next(3)
        let idx2 = (idx1 + 1 + rng.Next(2)) % 3

        // Pick random delta from -MaxDelta to +MaxDelta (excluding 0)
        let delta =
            let d = rng.NextDouble() * config.MaxDelta
            let d = if d < 1.0 then 1.0 else d  // Ensure minimum movement
            if rng.NextDouble() < 0.5 then -d else d

        let ep1 = state.[idx1]
        let ep2 = state.[idx2]

        let newDur1 = ep1.Duration + delta
        let newDur2 = ep2.Duration - delta

        // Check validity
        if newDur1 >= config.MinDuration && newDur2 >= config.MinDuration then
            let newState = Array.copy state
            newState.[idx1] <- { ep1 with Duration = newDur1 }
            newState.[idx2] <- { ep2 with Duration = newDur2 }
            Some newState
        else
            None

    /// Create initial state for a given total duration
    let initialState (totalDuration: float) : State =
        // Default split: 60/270/60 ratio scaled to total duration
        let ratio = totalDuration / 390.0
        [|
            { Label = Morning; Duration = 60.0 * ratio }
            { Label = Mid; Duration = 270.0 * ratio }
            { Label = Close; Duration = 60.0 * ratio }
        |]

    /// Sample session subdivision for a day
    let sample
        (config: Config)
        (mcmcConfig: MCMC.Config)
        (rng: Random)
        (dayDuration: float)
        : State =

        let initial = initialState dayDuration
        MCMC.run mcmcConfig (logLikelihood config) (propose config) initial rng

// =============================================================================
// Trend Level (Session -> Trends)
// =============================================================================

module TrendLevel =
    type Params = {
        DurationMean: float
        DurationStdDev: float
    }

    type Config = {
        /// Selection weights for each trend type, keyed by parent session
        SelectionWeights: Map<DaySession, Map<Trend, float>>
        /// Duration parameters for each trend type
        DurationParams: Map<Trend, Params>
        MinDuration: float
        MaxDelta: float
        MinTrends: int
    }

    let private defaultSelectionWeights : Map<DaySession, Map<Trend, float>> =
        let morningCloseWeights = Map.ofList [
            StrongUptrend, 0.15
            MidUptrend, 0.15
            WeakUptrend, 0.10
            Consolidation, 0.20
            WeakDowntrend, 0.10
            MidDowntrend, 0.15
            StrongDowntrend, 0.15
        ]
        let midWeights = Map.ofList [
            StrongUptrend, 0.02
            MidUptrend, 0.08
            WeakUptrend, 0.15
            Consolidation, 0.50
            WeakDowntrend, 0.15
            MidDowntrend, 0.08
            StrongDowntrend, 0.02
        ]
        Map.ofList [
            Morning, morningCloseWeights
            Mid, midWeights
            Close, morningCloseWeights
        ]

    let private defaultDurationParams : Map<Trend, Params> =
        Map.ofList [
            StrongUptrend, { DurationMean = 5.0; DurationStdDev = 2.0 }
            MidUptrend, { DurationMean = 15.0; DurationStdDev = 5.0 }
            WeakUptrend, { DurationMean = 30.0; DurationStdDev = 10.0 }
            Consolidation, { DurationMean = 20.0; DurationStdDev = 10.0 }
            WeakDowntrend, { DurationMean = 30.0; DurationStdDev = 10.0 }
            MidDowntrend, { DurationMean = 15.0; DurationStdDev = 5.0 }
            StrongDowntrend, { DurationMean = 5.0; DurationStdDev = 2.0 }
        ]

    let defaultConfig = {
        SelectionWeights = defaultSelectionWeights
        DurationParams = defaultDurationParams
        MinDuration = 1.0
        MaxDelta = 5.0
        MinTrends = 1
    }

    /// State for trend-level MCMC
    type State = Episode<Trend>[]

    /// Sample a trend type based on selection weights
    let sampleTrendType (weights: Map<Trend, float>) (rng: Random) : Trend =
        let trends, probs = weights |> Map.toArray |> Array.unzip
        let dist = Categorical(probs, rng)
        trends.[dist.Sample()]

    /// Compute log-likelihood for a trend state
    let logLikelihood (config: Config) (parentSession: DaySession) (state: State) : float =
        let weights = config.SelectionWeights.[parentSession]
        state
        |> Array.sumBy (fun ep ->
            // Log-likelihood of selecting this trend type
            let selectionLL = log(weights.[ep.Label])
            // Log-likelihood of this duration
            let durationParams = config.DurationParams.[ep.Label]
            let durationLL = Distribution.logNormalLogLikelihood durationParams.DurationMean durationParams.DurationStdDev ep.Duration
            selectionLL + durationLL)

    /// Propose a move: either adjust boundary or change trend type
    let propose (config: Config) (parentSession: DaySession) (rng: Random) (state: State) : State option =
        if state.Length < 2 then
            None
        else
            let moveType = rng.NextDouble()

            if moveType < 0.7 then
                // Pick two trends to transfer duration between
                let n = state.Length
                let idx1 = rng.Next(n)
                let idx2 = (idx1 + 1 + rng.Next(n - 1)) % n

                let delta =
                    let d = rng.NextDouble() * config.MaxDelta
                    let d = if d < 0.5 then 0.5 else d
                    if rng.NextDouble() < 0.5 then -d else d

                let ep1 = state.[idx1]
                let ep2 = state.[idx2]

                let newDur1 = ep1.Duration + delta
                let newDur2 = ep2.Duration - delta

                if newDur1 >= config.MinDuration && newDur2 >= config.MinDuration then
                    let newState = Array.copy state
                    newState.[idx1] <- { ep1 with Duration = newDur1 }
                    newState.[idx2] <- { ep2 with Duration = newDur2 }
                    Some newState
                else
                    None
            else
                // Change a trend type (sample uniformly for symmetric proposal)
                let idx = rng.Next(state.Length)
                let ep = state.[idx]
                let allTrends = config.DurationParams |> Map.keys |> Seq.toArray
                let newTrend = allTrends.[rng.Next(allTrends.Length)]

                let newState = Array.copy state
                newState.[idx] <- { ep with Label = newTrend }
                Some newState

    /// Create initial state for a given session duration
    let initialState (config: Config) (parentSession: DaySession) (rng: Random) (sessionDuration: float) : State =
        let weights = config.SelectionWeights.[parentSession]

        // Compute weighted expected duration based on selection probabilities
        let expectedDuration =
            weights
            |> Map.toSeq
            |> Seq.sumBy (fun (trend, weight) ->
                let p = config.DurationParams.[trend]
                weight * p.DurationMean)

        // Sample number of trends with some randomness
        let expectedNumTrends = sessionDuration / expectedDuration
        let numTrends =
            let baseCount = max config.MinTrends (int (round expectedNumTrends))
            let variation = rng.Next(-2, 3) // -2 to +2
            max config.MinTrends (baseCount + variation)

        // Generate trends with sampled types and durations from their distributions
        let trends =
            [| for _ in 1 .. numTrends ->
                let trend = sampleTrendType weights rng
                let p = config.DurationParams.[trend]
                let mu, sigma = Distribution.logNormalParams p.DurationMean p.DurationStdDev
                let duration = LogNormal(mu, sigma, rng).Sample()
                { Label = trend; Duration = duration } |]

        // Scale durations to sum exactly to sessionDuration
        let total = trends |> Array.sumBy (fun t -> t.Duration)
        let scale = sessionDuration / total
        trends |> Array.map (fun t -> { t with Duration = t.Duration * scale })

    /// Sample trend subdivision for a session
    let sample
        (config: Config)
        (mcmcConfig: MCMC.Config)
        (rng: Random)
        (parentSession: DaySession)
        (sessionDuration: float)
        : State =

        let initial = initialState config parentSession rng sessionDuration
        let ll = logLikelihood config parentSession
        let prop = propose config parentSession
        MCMC.run mcmcConfig ll prop initial rng

// =============================================================================
// Composition: Full Day Generation
// =============================================================================

/// Result of generating a full day of episodes
type DayResult = {
    Sessions: Episode<DaySession>[]
    Trends: Map<int, Episode<Trend>[]>  // Keyed by session index
}

/// Generate a complete day with sessions and trends
let generateDay
    (sessionConfig: SessionLevel.Config)
    (trendConfig: TrendLevel.Config)
    (mcmcConfig: MCMC.Config)
    (rng: Random)
    (dayDuration: float)
    : DayResult =

    // Level 1: Sample sessions
    let sessions = SessionLevel.sample sessionConfig mcmcConfig rng dayDuration

    // Level 2: Sample trends for each session
    let trends =
        sessions
        |> Array.mapi (fun i session ->
            let trendEpisodes = TrendLevel.sample trendConfig mcmcConfig rng session.Label session.Duration
            (i, trendEpisodes))
        |> Map.ofArray

    { Sessions = sessions; Trends = trends }

// =============================================================================
// Display Utilities
// =============================================================================

let printEpisodes (label: string) (episodes: Episode<'a>[]) (showLabel: 'a -> string) : unit =
    let total = episodes |> Array.sumBy (fun e -> e.Duration)
    printfn "%s (total %.1f):" label total
    let mutable t = 0.0
    for ep in episodes do
        printfn "  %6.1f - %6.1f: %-12s (%.1f)" t (t + ep.Duration) (showLabel ep.Label) ep.Duration
        t <- t + ep.Duration

let showSession (s: DaySession) : string =
    match s with
    | Morning -> "Morning"
    | Mid -> "Mid"
    | Close -> "Close"

let showTrend (t: Trend) : string =
    match t with
    | StrongUptrend -> "StrongUp"
    | MidUptrend -> "MidUp"
    | WeakUptrend -> "WeakUp"
    | Consolidation -> "Consol"
    | WeakDowntrend -> "WeakDown"
    | MidDowntrend -> "MidDown"
    | StrongDowntrend -> "StrongDown"

let printDayResult (result: DayResult) : unit =
    printEpisodes "Sessions" result.Sessions showSession
    printfn ""
    for i in 0 .. result.Sessions.Length - 1 do
        let session = result.Sessions.[i]
        let trends = result.Trends.[i]
        printEpisodes (sprintf "%s Trends" (showSession session.Label)) trends showTrend
        printfn ""
