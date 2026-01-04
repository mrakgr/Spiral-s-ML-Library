module Spiral.Trading.Simulation.OrderBook

open System
open MathNet.Numerics.Distributions

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
    TickSize: float            // e.g., 0.01
    BidMeanDistance: float     // Mean distance from midpoint for bids
    AskMeanDistance: float     // Mean distance from midpoint for asks
    LevelCount: int            // Number of levels to generate per side
    SizeMean: float            // Mean order size
    SizeStdDev: float          // Standard deviation of order size
}

module ParamConversion =
    /// Convert mean distance to exponential lambda
    let distanceToLambda (meanDistance: float) : float =
        1.0 / meanDistance
    
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
    (midpoint: float) 
    (tickSize: float) 
    (lambda: float) 
    (sizeMu: float) 
    (sizeSigma: float) 
    (levelCount: int) 
    (side: Side) 
    (rng: Random) : Level[] =
    
    // Create MathNet distributions
    let expDist = Exponential(lambda, rng)
    let logNormalDist = LogNormal(sizeMu, sizeSigma, rng)
    
    // Sample distances and create levels
    let levels = 
        Array.init levelCount (fun _ ->
            let distance = expDist.Sample()
            let size = logNormalDist.Sample()
            let rawPrice = 
                match side with
                | Bid -> midpoint - distance
                | Ask -> midpoint + distance
            let price = snapToTick rng tickSize rawPrice
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
    let bidLambda = ParamConversion.distanceToLambda config.BidMeanDistance
    let askLambda = ParamConversion.distanceToLambda config.AskMeanDistance
    let (sizeMu, sizeSigma) = ParamConversion.sizeToLogNormalParams config.SizeMean config.SizeStdDev
    
    let bids = generateSideLevels 
                config.Midpoint config.TickSize bidLambda 
                sizeMu sizeSigma config.LevelCount 
                Bid rng
    
    let asks = generateSideLevels 
                (config.Midpoint + config.TickSize) config.TickSize askLambda 
                sizeMu sizeSigma config.LevelCount 
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
