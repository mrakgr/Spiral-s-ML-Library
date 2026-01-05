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

type LimitParams = {
    Limit: float           // Best bid or best ask
    SizeMean: float        // Mean order size
    SizeStdDev: float      // Standard deviation of order size
}

type DistanceParams = {
    DistanceMean: float    // Mean distance from limit
    DistanceStdDev: float  // Std dev of distance from limit
}

type SideParams = {
    LevelCount: int        // Number of levels to generate
    LevelParams: LimitParams
    DistanceParams: DistanceParams
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
    
    let limitParams = sideParams.LevelParams
    let distParams = sideParams.DistanceParams
    
    // Create distributions
    let (shape, rate) = ParamConversion.distanceToGammaParams distParams.DistanceMean distParams.DistanceStdDev
    let distDist = Gamma(shape, rate, rng)
    let (mu, sigma) = ParamConversion.sizeToLogNormalParams limitParams.SizeMean limitParams.SizeStdDev
    let sizeDist = LogNormal(mu, sigma, rng)
    
    // Sample distances and create levels (always include one at distance 0)
    let levels = 
        Array.init sideParams.LevelCount (fun i ->
            let distance = if i = 0 then 0.0 else distDist.Sample()
            let rawSize = sizeDist.Sample()
            let size = stochasticRound rng rawSize |> int
            
            let rawPrice = 
                match side with
                | Bid -> limitParams.Limit - distance
                | Ask -> limitParams.Limit + distance
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
    
    { BestBid = config.Bid.LevelParams.Limit; BestAsk = config.Ask.LevelParams.Limit; Bids = bids; Asks = asks }

/// Generate just the BBO (best bid/offer) levels
let generateBBO (bidLimit: LimitParams) (askLimit: LimitParams) (tickSize: float) (rng: Random) : Level * Level =
    let distParams = { DistanceMean = 1.0; DistanceStdDev = 1.0 }
    let bidParams = { LevelCount = 1; LevelParams = bidLimit; DistanceParams = distParams }
    let askParams = { LevelCount = 1; LevelParams = askLimit; DistanceParams = distParams }
    let bid = (generateSideLevels bidParams tickSize Bid rng).[0]
    let ask = (generateSideLevels askParams tickSize Ask rng).[0]
    (bid, ask)

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
