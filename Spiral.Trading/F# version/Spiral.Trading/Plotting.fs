module Spiral.Trading.Plotting

open System
open System.IO
open XPlot.Plotly

/// Generate a candlestick chart with volume and save as HTML
let generateCandlestickChart (prices: Database.SplitAdjustedPriceRow array) (ticker: string) (outputPath: string) (width: int) (height: int) : unit =
    let dates = prices |> Array.map (fun p -> p.date)
    
    let candlestick =
        Candlestick(
            x = dates,
            ``open`` = (prices |> Array.map (fun p -> p.adj_open)),
            high = (prices |> Array.map (fun p -> p.adj_high)),
            low = (prices |> Array.map (fun p -> p.adj_low)),
            close = (prices |> Array.map (fun p -> p.adj_close)),
            name = ticker
        )
    
    let volume =
        Bar(
            x = dates,
            y = (prices |> Array.map (fun p -> p.adj_volume)),
            name = "Volume",
            marker = Marker(color = "rgba(100, 100, 200, 0.5)"),
            yaxis = "y2"
        )
    
    let layout =
        Layout(
            title = $"Daily Price Chart - {ticker} (Split Adjusted)",
            xaxis = Xaxis(title = "Date"),
            yaxis = Yaxis(title = "Price", domain = [| 0.3; 1.0 |]),
            yaxis2 = Yaxis(title = "Volume", domain = [| 0.0; 0.25 |]),
            width = width,
            height = height
        )
    
    let chart =
        [candlestick :> Trace; volume :> Trace]
        |> Chart.Plot
        |> Chart.WithLayout(layout)
    
    let html = chart.GetHtml()
    File.WriteAllText(outputPath, html)
    printfn "Chart saved to %s" outputPath

/// Generate a stock chart with split-adjusted prices from SQL
let generateChart (dbPath: string) (ticker: string) (outputPath: string) (width: int) (height: int) : unit =
    if not (File.Exists dbPath) then
        failwithf "Database not found at %s" dbPath
    
    use connection = Database.openConnection dbPath
    
    let prices = Database.getSplitAdjustedPricesByTicker connection ticker
    
    if prices.Length = 0 then
        printfn "No data found for ticker %s" ticker
    else
        printfn "Found %d records for %s (split-adjusted via SQL)" prices.Length ticker
        generateCandlestickChart prices ticker outputPath width height

/// Generate a DOM indicator chart against SPY
let generateDomChart (dbPath: string) (outputPath: string) (width: int) (height: int) : unit =
    if not (File.Exists dbPath) then
        failwithf "Database not found at %s" dbPath
    
    use connection = Database.openConnection dbPath
    Database.initializeSchema connection
    
    let domData = Database.getDomIndicator connection
    
    if domData.Length = 0 then
        printfn "No DOM indicator data found"
    else
        let dates = domData |> Array.map (fun d -> d.date)
        
        // Calculate cumulative DOM
        let mutable cumDom = 0.0
        let domValues = 
            domData 
            |> Array.map (fun d -> 
                cumDom <- cumDom + d.dom_contribution * 100.0
                cumDom)
        
        // Get SPY prices for the same date range
        let spyPrices = Database.getSplitAdjustedPricesByTicker connection "SPY"
        let domDateSet = Set.ofArray dates
        let spyFiltered = spyPrices |> Array.filter (fun p -> domDateSet.Contains p.date)
        
        let domTrace =
            Scatter(
                x = dates,
                y = domValues,
                name = "DOM",
                yaxis = "y"
            )
        
        let spyTrace =
            Scatter(
                x = (spyFiltered |> Array.map (fun p -> p.date)),
                y = (spyFiltered |> Array.map (fun p -> p.adj_close)),
                name = "SPY",
                yaxis = "y2"
            )
        
        let layout =
            Layout(
                title = "DOM Indicator vs SPY",
                xaxis = Xaxis(title = "Date"),
                yaxis = Yaxis(title = "DOM (cumulative)", side = "left"),
                yaxis2 = Yaxis(title = "SPY Price", side = "right", overlaying = "y"),
                width = width,
                height = height
            )
        
        let chart =
            [domTrace :> Trace; spyTrace :> Trace]
            |> Chart.Plot
            |> Chart.WithLayout layout
        
        let html = chart.GetHtml()
        File.WriteAllText(outputPath, html)
        printfn "DOM chart saved to %s" outputPath
        printfn "Date range: %s to %s (%d days)" dates[0] dates[dates.Length - 1] dates.Length
