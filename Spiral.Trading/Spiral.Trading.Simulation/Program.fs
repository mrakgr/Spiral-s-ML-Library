open System
open Spiral.Trading.Simulation.OrderBook

[<EntryPoint>]
let main argv =
    let rng = Random(42)

    let bidParams = {
        Limit = 99.95
        DistanceMean = 0.10
        DistanceStdDev = 0.15
        LevelCount = 20
        SizeMean = 200.0
        SizeStdDev = 150.0
    }

    let askParams = {
        Limit = 100.00
        DistanceMean = 0.10
        DistanceStdDev = 0.15
        LevelCount = 20
        SizeMean = 200.0
        SizeStdDev = 150.0
    }

    let config = {
        TickSize = 0.01
        Bid = bidParams
        Ask = askParams
    }

    let book = generate config rng
    print book
    
    0
