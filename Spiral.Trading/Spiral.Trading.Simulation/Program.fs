open System
open Spiral.Trading.Simulation.OrderBook

[<EntryPoint>]
let main argv =
    let rng = Random(42)

    let config = {
        Midpoint = 100.0
        TickSize = 0.01
        BidLambda = 2.0
        AskLambda = 1.5
        LevelCount = 20
        SizeMu = 5.0
        SizeSigma = 1.0
    }

    let book = generate config rng
    print book
    
    0
