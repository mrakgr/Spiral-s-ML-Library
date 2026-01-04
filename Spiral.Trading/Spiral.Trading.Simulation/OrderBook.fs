module Spiral.Trading.Simulation.OrderBook

open System

type Side = Bid | Ask

type Level = {
    Price: float
    Size: float
}

type OrderBook = {
    Midpoint: float
    Bids: Level[]  // Sorted descending (best bid first)
    Asks: Level[]  // Sorted ascending (best ask first)
}

type OrderBookParams = {
    Midpoint: float
    TickSize: float        // e.g., 0.01
    BidLambda: float       // Exponential rate for bid side
    AskLambda: float       // Exponential rate for ask side
    LevelCount: int        // Number of levels to generate per side
    SizeMu: float          // Log-normal mu for size
    SizeSigma: float       // Log-normal sigma for size
}

module Distributions =
    /// Sample from exponential distribution
    let sampleExponential (lambda: float) (rng: Random) : float =
        -log(1.0 - rng.NextDouble()) / lambda
    
    /// Sample from log-normal distribution
    let sampleLogNormal (mu: float) (sigma: float) (rng: Random) : float =
        let u1 = rng.NextDouble()
        let u2 = rng.NextDouble()
        // Box-Muller transform for standard normal
        let z = sqrt(-2.0 * log(u1)) * cos(2.0 * Math.PI * u2)
        exp(mu + sigma * z)

/// Snap a price to the nearest tick
let snapToTick (tickSize: float) (price: float) : float =
    round(price / tickSize) * tickSize

/// Generate order book levels for one side
let generateSideLevels 
    (midpoint: float) 
    (tickSize: float) 
    (lambda: float) 
    (sizeMu: float) 
    (sizeSigma: float) 
    (levelCount: int) 
    (side: Side) 
    (rng: Random) : Level[] =
    
    // Sample distances and create levels
    let levels = 
        Array.init levelCount (fun _ ->
            let distance = Distributions.sampleExponential lambda rng
            let size = Distributions.sampleLogNormal sizeMu sizeSigma rng
            let rawPrice = 
                match side with
                | Bid -> midpoint - distance
                | Ask -> midpoint + distance
            let price = snapToTick tickSize rawPrice
            { Price = price; Size = size }
        )
    
    // Group by price and sum sizes
    let aggregated = 
        levels
        |> Array.groupBy (fun l -> l.Price)
        |> Array.map (fun (price, lvls) -> 
            { Price = price; Size = lvls |> Array.sumBy (fun l -> l.Size) })
    
    // Sort appropriately
    match side with
    | Bid -> aggregated |> Array.sortByDescending (fun l -> l.Price)
    | Ask -> aggregated |> Array.sortBy (fun l -> l.Price)

/// Generate a complete order book
let generate (config: OrderBookParams) (rng: Random) : OrderBook =
    let bids = generateSideLevels 
                config.Midpoint config.TickSize config.BidLambda 
                config.SizeMu config.SizeSigma config.LevelCount 
                Bid rng
    
    let asks = generateSideLevels 
                config.Midpoint config.TickSize config.AskLambda 
                config.SizeMu config.SizeSigma config.LevelCount 
                Ask rng
    
    { Midpoint = config.Midpoint; Bids = bids; Asks = asks }

/// Pretty print an order book
let print (book: OrderBook) : unit =
    printfn "Order Book (Midpoint: %.2f)" book.Midpoint
    printfn ""
    printfn "%-12s %10s" "ASKS" ""
    printfn "%-12s %10s" "Price" "Size"
    printfn "%s" (String.replicate 24 "-")
    
    // Print asks in reverse (highest first for visual)
    for level in book.Asks |> Array.rev do
        printfn "%-12.2f %10.0f" level.Price level.Size
    
    printfn "%s" (String.replicate 24 "=")
    
    // Print bids
    for level in book.Bids do
        printfn "%-12.2f %10.0f" level.Price level.Size
    
    printfn "%s" (String.replicate 24 "-")
    printfn "%-12s %10s" "BIDS" ""
