module Spiral.Trading.Simulation.OrderBook

open System
open MathNet.Numerics.Distributions

type Side = Bid | Ask

type Level = {
    Price: float
    Size: float
}

type OrderBook = {
    BestBid: float
    BestAsk: float
    Bids: Level[]  // Sorted descending (best bid first)
    Asks: Level[]  // Sorted ascending (best ask first)
}

type OrderBookParams = {
    BestBid: float             // Hard limit for bids (can't exceed this)
    BestAsk: float             // Hard limit for asks (can't go below this)
    TickSize: float            // e.g., 0.01
    BidDistanceMean: float     // Mean distance from best bid for bid levels
    BidDistanceStdDev: float   // Std dev of distance from best bid
    AskDistanceMean: float     // Mean distance from best ask for ask levels
    AskDistanceStdDev: float   // Std dev of distance from best ask
    LevelCount: int            // Number of levels to generate per side
    SizeMean: float            // Mean order size
    SizeStdDev: float          // Standard deviation of order size
}

module ParamConversion =
    /// Convert mean and std dev to gamma shape and rate
    /// For gamma: mean = shape/rate, variance = shape/rate²
    let distanceToGammaParams (mean: float) (stdDev: float) : float * float =
        let variance = stdDev * stdDev
        let shape = (mean * mean) / variance
        let rate = mean / variance
        (shape, rate)
    
    /// Convert mean and std dev to log-normal mu and sigma
    /// For log-normal: mean = exp(mu + sigma²/2), variance = (exp(sigma²) - 1) * exp(2*mu + sigma²)
    let sizeToLogNormalParams (mean: float) (stdDev: float) : float * float =
        let variance = stdDev * stdDev
        let sigma2 = log(1.0 + variance / (mean * mean))
        let sigma = sqrt(sigma2)
        let mu = log(mean) - sigma2 / 2.0
        (mu, sigma)

/// Stochastic rounding: rounds up or down probabilistically based on fractional part
let stochasticRound (rng: Random) (x: float) : float =
    let floor = Math.Floor(x)
    let frac = x - floor
    if rng.NextDouble() < frac then floor + 1.0 else floor

/// Snap a price to tick size using stochastic rounding
let snapToTick (rng: Random) (tickSize: float) (price: float) : float =
    let ticks = price / tickSize
    let roundedTicks = stochasticRound rng ticks
    roundedTicks * tickSize

/// Generate order book levels for one side
let generateSideLevels 
    (limit: float)           // Best bid or best ask (hard limit)
    (tickSize: float) 
    (distanceMean: float) 
    (distanceStdDev: float)
    (sizeMean: float)
    (sizeStdDev: float)
    (levelCount: int) 
    (side: Side) 
    (rng: Random) : Level[] =
    
    // Create distributions
    let (shape, rate) = ParamConversion.distanceToGammaParams distanceMean distanceStdDev
    let distDist = Gamma(shape, rate, rng)
    let (mu, sigma) = ParamConversion.sizeToLogNormalParams sizeMean sizeStdDev
    let sizeDist = LogNormal(mu, sigma, rng)
    
    // Sample distances and create levels
    let levels = 
        Array.init levelCount (fun _ ->
            let distance = distDist.Sample()
            let size = sizeDist.Sample()
            
            let rawPrice = 
                match side with
                | Bid -> limit - distance
                | Ask -> limit + distance
            let price = snapToTick rng tickSize rawPrice
            
            // Clamp to limit
            let clampedPrice = 
                match side with
                | Bid -> min price limit
                | Ask -> max price limit
            
            { Price = clampedPrice; Size = size }
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
                config.BestBid config.TickSize 
                config.BidDistanceMean config.BidDistanceStdDev
                config.SizeMean config.SizeStdDev
                config.LevelCount Bid rng
    
    let asks = generateSideLevels 
                config.BestAsk config.TickSize 
                config.AskDistanceMean config.AskDistanceStdDev
                config.SizeMean config.SizeStdDev
                config.LevelCount Ask rng
    
    { BestBid = config.BestBid; BestAsk = config.BestAsk; Bids = bids; Asks = asks }

/// Pretty print an order book
let print (book: OrderBook) : unit =
    printfn "Order Book (Best Bid: %.2f, Best Ask: %.2f)" book.BestBid book.BestAsk
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
