open System
open Spiral.Trading.Simulation.OrderBook

[<EntryPoint>]
let main argv =
    let rng = Random(42)

    let config = {
        Midpoint = 100.0
        TickSize = 0.01
        BidMeanDistance = 0.50    // Mean $0.50 away from midpoint
        AskMeanDistance = 0.67    // Mean $0.67 away from midpoint (wider asks)
        LevelCount = 20
        SizeMean = 200.0          // Mean order size of 200 shares
        SizeStdDev = 150.0        // Std dev of 150 shares
    }

    let book = generate config rng
    print book
    
    0
