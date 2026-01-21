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

    /// Pick two distinct random indices from 0 to n-1
    let pickTwoDistinctIndices (rng: Random) (n: int) : int * int =
        let idx1 = rng.Next(n)
        let idx2 = (idx1 + 1 + rng.Next(n - 1)) % n
        (idx1, idx2)

    /// Transfer duration between two episodes in a state array
    let transferDuration (rng: Random) (maxDelta: float) (state: Episode<'a>[]) : Episode<'a>[] =
        let (idx1, idx2) = pickTwoDistinctIndices rng state.Length

        let delta =
            let d = rng.NextDouble() * maxDelta
            if rng.NextDouble() < 0.5 then -d else d

        let newState = Array.copy state
        newState.[idx1] <- { state.[idx1] with Duration = state.[idx1].Duration + delta }
        newState.[idx2] <- { state.[idx2] with Duration = state.[idx2].Duration - delta }
        newState

    /// Change the label of a random episode to a uniformly sampled label from the given array
    let changeLabel (rng: Random) (allLabels: 'a[]) (state: Episode<'a>[]) : Episode<'a>[] =
        let idx = rng.Next(state.Length)
        let newLabel = allLabels.[rng.Next(allLabels.Length)]

        let newState = Array.copy state
        newState.[idx] <- { state.[idx] with Label = newLabel }
        newState

    /// Sample a label from weighted options
    let sampleWeighted (rng: Random) (weights: seq<'a * float>) : 'a =
        let labels, probs = weights |> Seq.toArray |> Array.unzip
        let dist = Categorical(probs, rng)
        labels.[dist.Sample()]

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
    /// Distribution specification for positive values
    type Params =
        | LogNormal of mean: float * stdDev: float

    /// Convert mean/stdDev to log-normal mu/sigma parameters
    let logNormalParams (mean: float) (stdDev: float) : float * float =
        let variance = stdDev * stdDev
        let sigma2 = log(1.0 + variance / (mean * mean))
        let sigma = sqrt(sigma2)
        let mu = log(mean) - sigma2 / 2.0
        (mu, sigma)

    /// Compute log-likelihood for a value under the given distribution
    let logLikelihood (dist: Params) (value: float) : float =
        match dist with
        | LogNormal (mean, stdDev) ->
            if value <= 0.0 then
                Double.NegativeInfinity
            else
                let (mu, sigma) = logNormalParams mean stdDev
                let d = MathNet.Numerics.Distributions.LogNormal(mu, sigma)
                d.DensityLn(value)

    /// Sample a value from the given distribution
    let sample (rng: Random) (dist: Params) : float =
        match dist with
        | LogNormal (mean, stdDev) ->
            let (mu, sigma) = logNormalParams mean stdDev
            MathNet.Numerics.Distributions.LogNormal(mu, sigma, rng).Sample()

// =============================================================================
// Session Level (Day -> Sessions)
// =============================================================================

module SessionLevel =
    type Config = {
        MorningParams: Distribution.Params
        MidParams: Distribution.Params
        CloseParams: Distribution.Params
        MaxDelta: float
    }

    let defaultConfig = {
        MorningParams = Distribution.LogNormal (60.0, 20.0)
        MidParams = Distribution.LogNormal (270.0, 40.0)
        CloseParams = Distribution.LogNormal (60.0, 20.0)
        MaxDelta = 10.0
    }

    /// State for session-level MCMC: array of (session, duration) pairs
    type State = Episode<DaySession>[]

    let private getParams (config: Config) (session: DaySession) : Distribution.Params =
        match session with
        | Morning -> config.MorningParams
        | Mid -> config.MidParams
        | Close -> config.CloseParams

    /// Compute log-likelihood for a session state
    let logLikelihood (config: Config) (state: State) : float =
        state
        |> Array.sumBy (fun ep ->
            let p = getParams config ep.Label
            Distribution.logLikelihood p ep.Duration)

    /// Propose a move by transferring duration between two sessions
    let propose (config: Config) (rng: Random) (state: State) : State option =
        Some (MCMC.transferDuration rng config.MaxDelta state)

    /// Create initial state for a given total duration
    let initialState (totalDuration: float) : State =
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
    type Config = {
        SelectionWeights: Map<DaySession, Map<Trend, float>>
        DurationParams: Map<Trend, Distribution.Params>
        MaxDelta: float
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

    let private defaultDurationParams : Map<Trend, Distribution.Params> =
        Map.ofList [
            StrongUptrend, Distribution.LogNormal (5.0, 2.0)
            MidUptrend, Distribution.LogNormal (15.0, 5.0)
            WeakUptrend, Distribution.LogNormal (30.0, 10.0)
            Consolidation, Distribution.LogNormal (20.0, 10.0)
            WeakDowntrend, Distribution.LogNormal (30.0, 10.0)
            MidDowntrend, Distribution.LogNormal (15.0, 5.0)
            StrongDowntrend, Distribution.LogNormal (5.0, 2.0)
        ]

    let defaultConfig = {
        SelectionWeights = defaultSelectionWeights
        DurationParams = defaultDurationParams
        MaxDelta = 5.0
    }

    /// State for trend-level MCMC
    type State = Episode<Trend>[]

    /// Sample a trend type based on selection weights
    let sampleTrendType (weights: Map<Trend, float>) (rng: Random) : Trend =
        MCMC.sampleWeighted rng (Map.toSeq weights)

    /// Compute log-likelihood for a trend state
    let logLikelihood (config: Config) (parentSession: DaySession) (state: State) : float =
        let weights = config.SelectionWeights.[parentSession]
        state
        |> Array.sumBy (fun ep ->
            let selectionLL = log(weights.[ep.Label])
            let durationParams = config.DurationParams.[ep.Label]
            let durationLL = Distribution.logLikelihood durationParams ep.Duration
            selectionLL + durationLL)

    /// Propose a move: transfer duration, change label, or swap
    let propose (config: Config) (rng: Random) (state: State) : State option =
        let transferDuration () = MCMC.transferDuration rng config.MaxDelta state
        let changeLabel () = 
            let allTrends = config.DurationParams |> Map.keys |> Seq.toArray
            MCMC.changeLabel rng allTrends state

        let proposals = [
            transferDuration, if state.Length >= 2 then 0.7 else 0.0
            changeLabel, 0.3
        ]

        let move = MCMC.sampleWeighted rng proposals
        Some (move ())

    /// Create initial state for a given session duration
    let initialState (config: Config) (parentSession: DaySession) (rng: Random) (sessionDuration: float) : State =
        let weights = config.SelectionWeights.[parentSession]

        // Sample trends until total duration exceeds session duration
        let rec sampleTrends acc total =
            if total >= sessionDuration then
                acc
            else
                let trend = sampleTrendType weights rng
                let p = config.DurationParams.[trend]
                let duration = Distribution.sample rng p
                sampleTrends ({ Label = trend; Duration = duration } :: acc) (total + duration)

        let trends = sampleTrends [] 0.0 |> List.toArray

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
        let prop = propose config
        MCMC.run mcmcConfig ll prop initial rng

// =============================================================================
// Composition: Full Day Generation
// =============================================================================

/// Result of generating a full day of episodes
type DayResult = {
    Sessions: Episode<DaySession>[]
    Trends: Episode<Trend>[][]
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
        |> Array.map (fun session ->
            TrendLevel.sample trendConfig mcmcConfig rng session.Label session.Duration)

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

let showSession (s: DaySession) : string = sprintf "%A" s

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
