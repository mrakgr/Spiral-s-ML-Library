open System
open Argu
open Spiral.Trading.Simulation.OrderBook
open Spiral.Trading.Simulation.Episode
open Spiral.Trading.Simulation.EpisodeMCMC

type OrderBookArgs =
    | [<AltCommandLine("-s")>] Seed of int
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Seed _ -> "Random seed for generation"

type SimulateDayArgs =
    | [<AltCommandLine("-s")>] Seed of int
    | [<AltCommandLine("-n")>] Runs of int
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Seed _ -> "Random seed for simulation"
            | Runs _ -> "Number of days to simulate"

type MCMCArgs =
    | [<AltCommandLine("-s")>] Seed of int
    | [<AltCommandLine("-n")>] Samples of int
    | [<AltCommandLine("-i")>] Iterations of int
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Seed _ -> "Random seed for MCMC"
            | Samples _ -> "Number of samples to generate"
            | Iterations _ -> "MCMC iterations per sample"

type Command =
    | [<CliPrefix(CliPrefix.None)>] Order_Book of ParseResults<OrderBookArgs>
    | [<CliPrefix(CliPrefix.None)>] Simulate_Day of ParseResults<SimulateDayArgs>
    | [<CliPrefix(CliPrefix.None)>] Mcmc_Sessions of ParseResults<MCMCArgs>
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Order_Book _ -> "Generate and display an order book"
            | Simulate_Day _ -> "Simulate a trading day's session structure (hazard rate)"
            | Mcmc_Sessions _ -> "Sample day sessions using MCMC"

let runOrderBook (args: ParseResults<OrderBookArgs>) =
    let seed = args.GetResult(OrderBookArgs.Seed, 42)
    let rng = Random(seed)

    let limitParams = { Limit = 99.95; SizeMean = 200.0; SizeStdDev = 150.0 }
    let distParams = { DistanceMean = 0.10; DistanceStdDev = 0.15 }

    let config = {
        TickSize = 0.01
        Bid = { LevelCount = 20; LevelParams = { limitParams with Limit = 99.95 }; DistanceParams = distParams }
        Ask = { LevelCount = 20; LevelParams = { limitParams with Limit = 100.00 }; DistanceParams = distParams }
    }

    let book = generate config rng
    print book

let runSimulateDay (args: ParseResults<SimulateDayArgs>) =
    let seed = args.GetResult(SimulateDayArgs.Seed, 42)
    let runs = args.GetResult(Runs, 1)
    let rng = Random(seed)

    for i in 1 .. runs do
        if runs > 1 then printfn "=== Day %d ===" i
        let sessions = simulateDay defaultDayParams rng
        let trends = simulateTrends sessions rng
        printDaySummary sessions
        printfn ""
        printTrendSummary trends
        if runs > 1 then printfn ""

let runMCMCSessions (args: ParseResults<MCMCArgs>) =
    let seed = args.GetResult(MCMCArgs.Seed, 42)
    let samples = args.GetResult(Samples, 5)
    let iterations = args.GetResult(Iterations, 10000)
    let rng = Random(seed)
    
    let config = { defaultMCMCConfig with Iterations = iterations }
    
    printfn "MCMC Session Sampling (iterations=%d)" iterations
    printfn ""
    
    for i in 1 .. samples do
        printfn "=== Sample %d ===" i
        let initial = initialState()
        let sampled = runMCMC config initial rng
        printState sampled
        printfn ""

[<EntryPoint>]
let main argv =
    let parser = ArgumentParser.Create<Command>(programName = "Spiral.Trading.Simulation")
    
    try
        let results = parser.ParseCommandLine(inputs = argv, raiseOnUsage = true)
        
        match results.GetSubCommand() with
        | Order_Book args -> runOrderBook args
        | Simulate_Day args -> runSimulateDay args
        | Mcmc_Sessions args -> runMCMCSessions args
        
        0
    with
    | :? ArguParseException as e ->
        printfn "%s" e.Message
        1
