#r "nuget: MathNet.Numerics, 5.0.0"

open System
open MathNet.Numerics.Distributions

// Baseline volatility: 1% per sqrt(1,000,000) shares
let baselineVol = 0.01 / sqrt(1_000_000.0)  // = 0.00001 per sqrt-share

// Trend strength ratios (drift/vol)
let getTrendRatio trend =
    match trend with
    | "StrongUptrend"   ->  0.5
    | "MidUptrend"      ->  0.3
    | "WeakUptrend"     ->  0.15
    | "Consolidation"   ->  0.0
    | "WeakDowntrend"   -> -0.15
    | "MidDowntrend"    -> -0.3
    | "StrongDowntrend" -> -0.5
    | _ -> 0.0

// Single trade price update
let applyTrade (rng: Random) (baseVol: float) (trendRatio: float) (size: int) (price: float) =
    let vol = baseVol * sqrt(float size)
    let drift = trendRatio * vol
    let z = Normal.Sample(rng, 0.0, 1.0)
    price * exp(drift - vol * vol / 2.0 + vol * z)

// Test: simulate 1000 trades of size 100 with different trends
let rng = Random(42)
let startPrice = 100.0

printfn "Baseline vol: %.6f%% per sqrt-share" (baselineVol * 100.0)
printfn "For 100 shares: vol = %.4f%%" (baselineVol * sqrt(100.0) * 100.0)
printfn ""

for trend in ["StrongUptrend"; "Consolidation"; "StrongDowntrend"] do
    let ratio = getTrendRatio trend
    let mutable price = startPrice
    let tradeSize = 100
    let numTrades = 1000
    
    for _ in 1 .. numTrades do
        price <- applyTrade rng baselineVol ratio tradeSize price
    
    let totalVol = baselineVol * sqrt(float tradeSize) * sqrt(float numTrades)
    let returnPct = (price - startPrice) / startPrice * 100.0
    printfn "%s (ratio=%.1f):" trend ratio
    printfn "  Final price: %.4f (%.2f%% return)" price returnPct
    printfn "  Expected vol over %d trades: %.2f%%" numTrades (totalVol * 100.0)
    printfn ""

// Test: same total volume, different trade sizes
printfn "=== Same total volume (10000 shares), different trade sizes ==="
let totalShares = 10000

for tradeSize in [10; 100; 1000] do
    let numTrades = totalShares / tradeSize
    let mutable price = startPrice
    let ratio = 0.0  // Consolidation
    
    for _ in 1 .. numTrades do
        price <- applyTrade rng baselineVol ratio tradeSize price
    
    let returnPct = (price - startPrice) / startPrice * 100.0
    printfn "  %d trades x %d shares: final=%.4f (%.2f%%)" numTrades tradeSize price returnPct
