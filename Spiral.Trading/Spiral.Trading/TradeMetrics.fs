module Spiral.Trading.TradeMetrics

open System
open System.Data
open System.Collections.Generic
open Dapper

/// Trade side classification
type TradeSide = Buy | Sell | Mid

/// Trade with quote data (from trades_with_quotes view)
[<CLIMutable>]
type TradeWithQuote = {
    id: int64
    ticker: string
    session_date: DateOnly
    sip_timestamp: DateTime
    participant_timestamp: DateTime
    sequence_number: int64
    price: float
    size: float
    exchange: int
    tape: Nullable<int>
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    side: string
}

/// Computed metrics for a trade
type TradeMetrics = {
    Id: int64
    Vwap: float
    Vwstd: float
    AskVolume: float
    BidVolume: float
    MidVolume: float
    TotalVolume: float
}

/// Running state for incremental metric computation
type private RunningState = {
    mutable SumPriceSize: float
    mutable SumPriceSqSize: float
    mutable SumSize: float
    mutable AskVolume: float
    mutable BidVolume: float
    mutable MidVolume: float
}

module private RunningState =
    let create () = {
        SumPriceSize = 0.0
        SumPriceSqSize = 0.0
        SumSize = 0.0
        AskVolume = 0.0
        BidVolume = 0.0
        MidVolume = 0.0
    }
    
    let add (state: RunningState) (trade: TradeWithQuote) =
        let ps = trade.price * trade.size
        state.SumPriceSize <- state.SumPriceSize + ps
        state.SumPriceSqSize <- state.SumPriceSqSize + trade.price * ps
        state.SumSize <- state.SumSize + trade.size
        match trade.side with
        | "BUY" -> state.AskVolume <- state.AskVolume + trade.size
        | "SELL" -> state.BidVolume <- state.BidVolume + trade.size
        | _ -> state.MidVolume <- state.MidVolume + trade.size
    
    let remove (state: RunningState) (trade: TradeWithQuote) =
        let ps = trade.price * trade.size
        state.SumPriceSize <- state.SumPriceSize - ps
        state.SumPriceSqSize <- state.SumPriceSqSize - trade.price * ps
        state.SumSize <- state.SumSize - trade.size
        match trade.side with
        | "BUY" -> state.AskVolume <- state.AskVolume - trade.size
        | "SELL" -> state.BidVolume <- state.BidVolume - trade.size
        | _ -> state.MidVolume <- state.MidVolume - trade.size
    
    let toMetrics (state: RunningState) (id: int64) : TradeMetrics =
        let vwap = if state.SumSize > 0.0 then state.SumPriceSize / state.SumSize else 0.0
        let vwstd = 
            if state.SumSize > 0.0 then
                let variance = state.SumPriceSqSize / state.SumSize - vwap * vwap
                sqrt (max 0.0 variance)
            else 0.0
        {
            Id = id
            Vwap = vwap
            Vwstd = vwstd
            AskVolume = state.AskVolume
            BidVolume = state.BidVolume
            MidVolume = state.MidVolume
            TotalVolume = state.SumSize
        }

/// Compute trade metrics with sliding window
/// windowSeconds: time window in seconds (trades ordered by sip_timestamp, filtered by participant_timestamp)
let computeMetrics (trades: TradeWithQuote[]) (windowSeconds: float) : TradeMetrics[] =
    if trades.Length = 0 then [||]
    else
        let window = TimeSpan.FromSeconds(windowSeconds)
        let results = Array.zeroCreate<TradeMetrics> trades.Length
        let state = RunningState.create ()
        
        // Priority queue keyed by participant_timestamp (min-heap)
        let pq = PriorityQueue<int, DateTime>()
        
        for i = 0 to trades.Length - 1 do
            let trade = trades[i]
            let windowStart = trade.sip_timestamp - window
            
            // Remove trades whose participant_timestamp is outside the window
            while pq.Count > 0 && trades[pq.Peek()].participant_timestamp < windowStart do
                let idx = pq.Dequeue()
                RunningState.remove state trades[idx]
            
            // Add current trade to state and priority queue
            RunningState.add state trade
            pq.Enqueue(i, trade.participant_timestamp)
            
            results[i] <- RunningState.toMetrics state trade.id
        
        results

/// Load trades from database for a specific ticker and session
let loadTrades (connection: IDbConnection) (ticker: string) (sessionDate: DateOnly) : TradeWithQuote[] =
    let sql = """
        SELECT 
            id, ticker, session_date, sip_timestamp, participant_timestamp,
            sequence_number, price, size, exchange, tape,
            bid_price, ask_price, bid_size, ask_size, side
        FROM trades_with_quotes 
        WHERE ticker = $ticker AND session_date = $session_date
        ORDER BY sip_timestamp
    """
    let parameters = {| ticker = ticker; session_date = sessionDate.ToString("yyyy-MM-dd") |}
    connection.Query<TradeWithQuote>(sql, parameters) |> Seq.toArray

/// Compute metrics for a specific ticker and session
let computeForSession 
    (connection: IDbConnection) 
    (ticker: string) 
    (sessionDate: DateOnly) 
    (windowSeconds: float) : TradeMetrics[] =
    let trades = loadTrades connection ticker sessionDate
    computeMetrics trades windowSeconds

/// Export trades with multi-window metrics to CSV
/// Windows are 2^n seconds where n is in [-1, 6] (0.5s, 1s, 2s, 4s, 8s, 16s, 32s, 64s)
let exportToCsv (connection: IDbConnection) (ticker: string) (sessionDate: DateOnly) (outputPath: string) =
    let trades = loadTrades connection ticker sessionDate
    if trades.Length = 0 then
        printfn "No trades found for %s on %s" ticker (sessionDate.ToString())
    else
        // Window sizes: 2^n for n in [-1, 6]
        let windowSizes = [| 0.5; 1.0; 2.0; 4.0; 8.0; 16.0; 32.0; 64.0 |]
        
        // Compute metrics for each window size
        let allMetrics = windowSizes |> Array.map (fun w -> computeMetrics trades w)
        
        use writer = new System.IO.StreamWriter(outputPath)
        
        // Write header
        let baseHeaders = "id,sip_timestamp,participant_timestamp,price,size,side,bid_price,ask_price,spread"
        let windowHeaders = 
            windowSizes 
            |> Array.map (fun w -> 
                let label = if w < 1.0 then "0.5s" else sprintf "%.0fs" w
                sprintf "vwap_%s,vwstd_%s,ask_pct_%s,bid_pct_%s,mid_pct_%s,vol_%s" label label label label label label)
            |> String.concat ","
        writer.WriteLine(baseHeaders + "," + windowHeaders)
        
        // Write data rows
        for i = 0 to trades.Length - 1 do
            let t = trades[i]
            let spread = t.ask_price - t.bid_price
            
            let baseCols = sprintf "%d,%s,%s,%.4f,%.0f,%s,%.4f,%.4f,%.4f"
                            t.id
                            (t.sip_timestamp.ToString("yyyy-MM-dd HH:mm:ss.ffffff"))
                            (t.participant_timestamp.ToString("yyyy-MM-dd HH:mm:ss.ffffff"))
                            t.price t.size t.side
                            t.bid_price t.ask_price spread
            
            let windowCols =
                allMetrics
                |> Array.map (fun metrics ->
                    let m = metrics[i]
                    let askPct = if m.TotalVolume > 0.0 then 100.0 * m.AskVolume / m.TotalVolume else 0.0
                    let bidPct = if m.TotalVolume > 0.0 then 100.0 * m.BidVolume / m.TotalVolume else 0.0
                    let midPct = if m.TotalVolume > 0.0 then 100.0 * m.MidVolume / m.TotalVolume else 0.0
                    sprintf "%.4f,%.4f,%.1f,%.1f,%.1f,%.0f" m.Vwap m.Vwstd askPct bidPct midPct m.TotalVolume)
                |> String.concat ","
            
            writer.WriteLine(baseCols + "," + windowCols)
        
        printfn "Exported %d trades to %s" trades.Length outputPath

/// Export simplified trade table for human reading
/// Sorted by participant_timestamp, 12 essential columns
let exportSimplified (connection: IDbConnection) (ticker: string) (sessionDate: DateOnly) (outputPath: string) =
    let trades = loadTrades connection ticker sessionDate
    if trades.Length = 0 then
        printfn "No trades found for %s on %s" ticker (sessionDate.ToString())
    else
        // Sort by participant_timestamp
        let sortedTrades = trades |> Array.sortBy (fun t -> t.participant_timestamp)
        
        // Compute 4s window metrics
        let metrics = computeMetrics sortedTrades 4.0
        
        use writer = new System.IO.StreamWriter(outputPath)
        
        // Write header
        writer.WriteLine("participant_timestamp,time_gap,price,size,side,bid_price,ask_price,vwap_4s,vwstd_4s,vol_4s,vol_delta_pct_4s,mid_pct_4s")
        
        // Write data rows
        for i = 0 to sortedTrades.Length - 1 do
            let t = sortedTrades[i]
            let m = metrics[i]
            
            // Time gap from previous trade
            let timeGap = 
                if i = 0 then 0.0
                else (t.participant_timestamp - sortedTrades[i-1].participant_timestamp).TotalSeconds
            
            // Volume percentages
            let askPct = if m.TotalVolume > 0.0 then 100.0 * m.AskVolume / m.TotalVolume else 0.0
            let bidPct = if m.TotalVolume > 0.0 then 100.0 * m.BidVolume / m.TotalVolume else 0.0
            let midPct = if m.TotalVolume > 0.0 then 100.0 * m.MidVolume / m.TotalVolume else 0.0
            let volDeltaPct = askPct - bidPct
            
            let row = sprintf "%s,%.3f,%.4f,%.0f,%s,%.4f,%.4f,%.4f,%.4f,%.0f,%.1f,%.1f"
                        (t.participant_timestamp.ToString("HH:mm:ss.ffffff"))
                        timeGap
                        t.price t.size t.side
                        t.bid_price t.ask_price
                        m.Vwap m.Vwstd m.TotalVolume
                        volDeltaPct midPct
            
            writer.WriteLine(row)
        
        printfn "Exported %d trades to %s" sortedTrades.Length outputPath
