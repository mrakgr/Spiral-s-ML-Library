module Spiral.Trading.Plotting

open System
open System.IO
open XPlot.Plotly

/// Adjust prices for stock splits (modifies array in place)
let adjustForSplits (prices: DailyPrice array) (splits: Split array) : unit =
    if splits.Length > 0 && prices.Length > 0 then
        let mutable currentMultiplier = 1.0
        let mutable splitIndex = 0
        
        // Iterate from newest to oldest (prices are sorted oldest to newest)
        for i = prices.Length - 1 downto 0 do
            let price = prices.[i]
            
            // Check if we passed any split dates (going backwards in time)
            while splitIndex < splits.Length && price.Date < splits.[splitIndex].ExecutionDate do
                let s = splits.[splitIndex]
                // If 1 share becomes 10 (10:1 split), From=1, To=10
                // Old price needs to be divided by 10
                currentMultiplier <- currentMultiplier * (s.SplitFrom / s.SplitTo)
                splitIndex <- splitIndex + 1
            
            if currentMultiplier <> 1.0 then
                prices.[i] <- {
                    price with
                        Open = decimal (float price.Open * currentMultiplier)
                        High = decimal (float price.High * currentMultiplier)
                        Low = decimal (float price.Low * currentMultiplier)
                        Close = decimal (float price.Close * currentMultiplier)
                        Volume = int64 (float price.Volume / currentMultiplier)
                }

/// Generate a candlestick chart and save as HTML
let generateCandlestickChart (prices: DailyPrice array) (ticker: string) (outputPath: string) (width: int) (height: int) : unit =
    let chart =
        Candlestick(
            x = (prices |> Array.map (fun p -> p.Date.ToString("yyyy-MM-dd"))),
            ``open`` = (prices |> Array.map (fun p -> float p.Open)),
            high = (prices |> Array.map (fun p -> float p.High)),
            low = (prices |> Array.map (fun p -> float p.Low)),
            close = (prices |> Array.map (fun p -> float p.Close)),
            name = ticker
        )
        |> Chart.Plot
        |> Chart.WithLayout(
            Layout(
                title = $"Daily Price Chart - {ticker}",
                xaxis = Xaxis(title = "Date"),
                yaxis = Yaxis(title = "Price"),
                width = width,
                height = height
            )
        )
    
    let html = chart.GetHtml()
    File.WriteAllText(outputPath, html)
    printfn "Chart saved to %s" outputPath

/// Generate a stock chart with split-adjusted prices
let generateChart (dbPath: string) (ticker: string) (outputPath: string) (width: int) (height: int) : unit =
    if not (File.Exists dbPath) then
        failwithf "Database not found at %s" dbPath
    
    use connection = Database.openConnection dbPath
    
    let prices = Database.getDailyPricesByTicker connection ticker
    
    if prices.Length = 0 then
        printfn "No data found for ticker %s" ticker
    else
        printfn "Found %d records for %s" prices.Length ticker
        
        let splits = Database.getSplitsByTicker connection ticker
        
        if splits.Length > 0 then
            printfn "Found %d splits for %s. Adjusting prices..." splits.Length ticker
            adjustForSplits prices splits
        
        generateCandlestickChart prices ticker outputPath width height
