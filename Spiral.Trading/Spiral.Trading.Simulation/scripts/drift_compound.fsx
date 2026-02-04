// Per-trade drift calculation
let driftPerSqrtShare = 0.001 / sqrt(1_000_000.0)  // 0.1% per sqrt(1M)
let avgSize = 200.0
let driftPerTrade = driftPerSqrtShare * sqrt(avgSize)

printfn "Drift per sqrt-share: %.10f" driftPerSqrtShare
printfn "Avg trade size: %.0f" avgSize
printfn "Drift per trade: %.8f%%" (driftPerTrade * 100.0)
printfn ""

// Over 600k trades (compounding)
let numTrades = 600_000
let compoundedReturn = (1.0 + driftPerTrade) ** float numTrades - 1.0
printfn "Trades per day: %d" numTrades
printfn "Compounded return: %.2f%%" (compoundedReturn * 100.0)
printfn ""

// What drift do we need for ~5% daily move?
let targetDailyReturn = 0.05
let neededPerTradeDrift = (1.0 + targetDailyReturn) ** (1.0 / float numTrades) - 1.0
printfn "For 5%% daily return:"
printfn "  Needed per-trade drift: %.12f%%" (neededPerTradeDrift * 100.0)
printfn "  That's %.2e per trade" neededPerTradeDrift
