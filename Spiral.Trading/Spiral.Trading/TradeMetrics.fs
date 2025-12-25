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
