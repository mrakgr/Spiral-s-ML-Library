open System
open Spiral.Trading.Simulation.OrderBook

[<EntryPoint>]
let main argv =
    let rng = Random(42)

    let config = {
        BestBid = 99.95
        BestAsk = 100.00
        TickSize = 0.01
        BidDistanceMean = 0.10     // Mean $0.10 away from best bid
        BidDistanceStdDev = 0.15   // Std dev of distance
        AskDistanceMean = 0.10     // Mean $0.10 away from best ask
        AskDistanceStdDev = 0.15   // Std dev of distance
        LevelCount = 20
        SizeMean = 200.0           // Mean order size of 200 shares
        SizeStdDev = 150.0         // Std dev of 150 shares
    }

    let book = generate config rng
    print book
    
    0
