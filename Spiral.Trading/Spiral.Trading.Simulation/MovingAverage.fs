module Spiral.Trading.Simulation.MovingAverage

open Spiral.Trading.Simulation.PriceGeneration

/// Trading position
type Position = Long | Short | Flat

/// Trade record
type Trade = {
    EntryTime: float
    EntryPrice: float
    ExitTime: float
    ExitPrice: float
    Position: Position
    PnL: float
}

/// Backtest result
type BacktestResult = {
    Trades: Trade[]
    TotalPnL: float
    WinRate: float
    NumTrades: int
}

/// Calculate simple moving average for an array of prices
let sma (prices: float[]) (period: int) : float[] =
    if prices.Length < period then
        [||]
    else
        let result = Array.zeroCreate (prices.Length - period + 1)
        let mutable sum = 0.0
        
        for i in 0 .. period - 1 do
            sum <- sum + prices.[i]
        result.[0] <- sum / float period
        
        for i in period .. prices.Length - 1 do
            sum <- sum + prices.[i] - prices.[i - period]
            result.[i - period + 1] <- sum / float period
        
        result

/// Run MA crossover backtest
let backtest (bars: Bar[]) (fastPeriod: int) (slowPeriod: int) : BacktestResult =
    let closes = bars |> Array.map (fun b -> b.Close)
    let fastMA = sma closes fastPeriod
    let slowMA = sma closes slowPeriod
    
    let offset = slowPeriod - 1
    let alignedFast = fastMA.[slowPeriod - fastPeriod ..]
    
    let trades = ResizeArray<Trade>()
    let mutable position = Flat
    let mutable entryTime = 0.0
    let mutable entryPrice = 0.0
    
    for i in 1 .. slowMA.Length - 1 do
        let barIdx = offset + i
        let bar = bars.[barIdx]
        let fast = alignedFast.[i]
        let slow = slowMA.[i]
        let prevFast = alignedFast.[i - 1]
        let prevSlow = slowMA.[i - 1]
        
        // Detect crossover
        let crossUp = prevFast <= prevSlow && fast > slow
        let crossDown = prevFast >= prevSlow && fast < slow
        
        match position, crossUp, crossDown with
        | Flat, true, _ ->
            position <- Long
            entryTime <- bar.Time
            entryPrice <- bar.Close
        | Flat, _, true ->
            position <- Short
            entryTime <- bar.Time
            entryPrice <- bar.Close
        | Long, _, true ->
            let pnl = bar.Close - entryPrice
            trades.Add({ EntryTime = entryTime; EntryPrice = entryPrice
                         ExitTime = bar.Time; ExitPrice = bar.Close
                         Position = Long; PnL = pnl })
            position <- Short
            entryTime <- bar.Time
            entryPrice <- bar.Close
        | Short, true, _ ->
            let pnl = entryPrice - bar.Close
            trades.Add({ EntryTime = entryTime; EntryPrice = entryPrice
                         ExitTime = bar.Time; ExitPrice = bar.Close
                         Position = Short; PnL = pnl })
            position <- Long
            entryTime <- bar.Time
            entryPrice <- bar.Close
        | _ -> ()
    
    let tradeArr = trades.ToArray()
    let totalPnL = tradeArr |> Array.sumBy (fun t -> t.PnL)
    let wins = tradeArr |> Array.filter (fun t -> t.PnL > 0.0) |> Array.length
    let winRate = if tradeArr.Length > 0 then float wins / float tradeArr.Length else 0.0
    
    { Trades = tradeArr; TotalPnL = totalPnL; WinRate = winRate; NumTrades = tradeArr.Length }

/// Print backtest results
let printResult (result: BacktestResult) (fastPeriod: int) (slowPeriod: int) : unit =
    printfn "MA Crossover Backtest (fast=%d, slow=%d):" fastPeriod slowPeriod
    printfn "  Trades: %d" result.NumTrades
    printfn "  Total PnL: %.4f" result.TotalPnL
    printfn "  Win Rate: %.1f%%" (result.WinRate * 100.0)
