module Spiral.Trading.Simulation.OrderBook

open System
open MathNet.Numerics.Distributions

type Side = Bid | Ask

type Level = {
    Price: float
    Size: int
}

type OrderBook = {
    BestBid: float
    BestAsk: float
    Bids: Level[]  // Sorted descending (best bid first)
    Asks: Level[]  // Sorted ascending (best ask first)
}

type SideParams = {
    Limit: float           // Best bid or best ask
    DistanceMean: float    // Mean distance from limit
    DistanceStdDev: float  // Std dev of distance from limit
    LevelCount: int        // Number of levels to generate
    SizeMean: float        // Mean order size
    SizeStdDev: float      // Standard deviation of order size
}

type OrderBookParams = {
    TickSize: float        // e.g., 0.01
    Bid: SideParams
    Ask: SideParams
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
    (sideParams: SideParams)
    (tickSize: float) 
    (side: Side) 
    (rng: Random) : Level[] =
    
    // Create distributions
    let (shape, rate) = ParamConversion.distanceToGammaParams sideParams.DistanceMean sideParams.DistanceStdDev
    let distDist = Gamma(shape, rate, rng)
    let (mu, sigma) = ParamConversion.sizeToLogNormalParams sideParams.SizeMean sideParams.SizeStdDev
    let sizeDist = LogNormal(mu, sigma, rng)
    
    // Sample distances and create levels (always include one at distance 0)
    let levels = 
        Array.init sideParams.LevelCount (fun i ->
            let distance = if i = 0 then 0.0 else distDist.Sample()
            let rawSize = sizeDist.Sample()
            let size = stochasticRound rng rawSize |> int
            
            let rawPrice = 
                match side with
                | Bid -> sideParams.Limit - distance
                | Ask -> sideParams.Limit + distance
            let price = snapToTick rng tickSize rawPrice
            
            { Price = price; Size = size }
        )
    
    // Group by price and sum sizes, filter out zero-size levels
    let aggregated = 
        levels
        |> Array.groupBy (fun l -> l.Price)
        |> Array.map (fun (price, lvls) -> 
            { Price = price; Size = lvls |> Array.sumBy (fun l -> l.Size) })
        |> Array.filter (fun l -> l.Size > 0)
    
    // Sort appropriately
    match side with
    | Bid -> aggregated |> Array.sortByDescending (fun l -> l.Price)
    | Ask -> aggregated |> Array.sortBy (fun l -> l.Price)

/// Generate a complete order book
let generate (config: OrderBookParams) (rng: Random) : OrderBook =
    let bids = generateSideLevels config.Bid config.TickSize Bid rng
    let asks = generateSideLevels config.Ask config.TickSize Ask rng
    
    { BestBid = config.Bid.Limit; BestAsk = config.Ask.Limit; Bids = bids; Asks = asks }

/// Pretty print an order book
let print (book: OrderBook) : unit =
    printfn "Order Book (Best Bid: %.2f, Best Ask: %.2f)" book.BestBid book.BestAsk
    printfn ""
    printfn "%-12s %10s" "ASKS" ""
    printfn "%-12s %10s" "Price" "Size"
    printfn "%s" (String.replicate 24 "-")
    
    // Print asks in reverse (highest first for visual)
    for level in book.Asks |> Array.rev do
        printfn "%-12.2f %10d" level.Price level.Size
    
    printfn "%s" (String.replicate 24 "=")
    
    // Print bids
    for level in book.Bids do
        printfn "%-12.2f %10d" level.Price level.Size
    
    printfn "%s" (String.replicate 24 "-")
    printfn "%-12s %10s" "BIDS" ""
