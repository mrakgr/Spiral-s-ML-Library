open System
open Spiral.Trading.Simulation.OrderBook

[<EntryPoint>]
let main argv =
    let rng = Random(42)

    let size = { SizeMean = 200.0; SizeStdDev = 150.0 }
    let dist = { DistanceMean = 0.10; DistanceStdDev = 0.15; Size = size }

    let config = {
        TickSize = 0.01
        Bid = { Limit = 99.95; LevelCount = 20; Distribution = dist }
        Ask = { Limit = 100.00; LevelCount = 20; Distribution = dist }
    }

    let book = generate config rng
    print book
    
    0
