open System
open Argu
open Spiral.Trading.Simulation.OrderBook
open Spiral.Trading.Simulation.EpisodeMCMC

type OrderBookArgs =
    | [<AltCommandLine("-s")>] Seed of int
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Seed _ -> "Random seed for generation"

type GenerateDayArgs =
    | [<AltCommandLine("-s")>] Seed of int
    | [<AltCommandLine("-n")>] Runs of int
    | [<AltCommandLine("-i")>] Iterations of int
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Seed _ -> "Random seed for generation"
            | Runs _ -> "Number of days to generate"
            | Iterations _ -> "MCMC iterations per level"

type Command =
    | [<CliPrefix(CliPrefix.None)>] Order_Book of ParseResults<OrderBookArgs>
    | [<CliPrefix(CliPrefix.None)>] Generate_Day of ParseResults<GenerateDayArgs>
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Order_Book _ -> "Generate and display an order book"
            | Generate_Day _ -> "Generate a full day with sessions and trends using MCMC"

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

let runGenerateDay (args: ParseResults<GenerateDayArgs>) =
    let seed = args.GetResult(GenerateDayArgs.Seed, 42)
    let runs = args.GetResult(Runs, 1)
    let iterations = args.GetResult(Iterations, 10000)
    let rng = Random(seed)

    let mcmcConfig = { MCMC.Iterations = iterations }
    let sessionConfig = SessionLevel.defaultConfig
    let trendConfig = TrendLevel.defaultConfig

    printfn "Day Generation (MCMC iterations=%d)" iterations
    printfn ""

    for i in 1 .. runs do
        if runs > 1 then printfn "=== Day %d ===" i
        let result = generateDay sessionConfig trendConfig mcmcConfig rng 390.0
        printDayResult result
        if i < runs then printfn "---"
        printfn ""

[<EntryPoint>]
let main argv =
    let parser = ArgumentParser.Create<Command>(programName = "Spiral.Trading.Simulation")

    try
        let results = parser.ParseCommandLine(inputs = argv, raiseOnUsage = true)

        match results.GetSubCommand() with
        | Order_Book args -> runOrderBook args
        | Generate_Day args -> runGenerateDay args

        0
    with
    | :? ArguParseException as e ->
        printfn "%s" e.Message
        1
