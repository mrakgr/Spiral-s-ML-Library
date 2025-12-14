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

/// Generate a DOM indicator chart against a reference ticker
let generateDomChart (dbPath: string) (ticker: string option) (outputPath: string) (width: int) (height: int) : unit =
    if not (File.Exists dbPath) then
        failwithf "Database not found at %s" dbPath
    
    let referenceTicker = ticker |> Option.defaultValue "SPY"
    
    use connection = Database.openConnection dbPath
    
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
        
        // Get reference ticker prices for the same date range
        let refPrices = Database.getSplitAdjustedPricesByTicker connection referenceTicker
        let domDateSet = Set.ofArray dates
        let refFiltered = refPrices |> Array.filter (fun p -> domDateSet.Contains p.date)
        
        let domTrace =
            Scatter(
                x = dates,
                y = domValues,
                name = "DOM",
                yaxis = "y"
            )
        
        let refTrace =
            Scatter(
                x = (refFiltered |> Array.map (fun p -> p.date)),
                y = (refFiltered |> Array.map (fun p -> p.adj_close)),
                name = referenceTicker,
                yaxis = "y2"
            )
        
        let layout =
            Layout(
                title = $"DOM Indicator vs {referenceTicker}",
                xaxis = Xaxis(title = "Date"),
                yaxis = Yaxis(title = "DOM (cumulative)", side = "left"),
                yaxis2 = Yaxis(title = $"{referenceTicker} Price", side = "right", overlaying = "y"),
                width = width,
                height = height
            )
        
        let chart =
            [domTrace :> Trace; refTrace :> Trace]
            |> Chart.Plot
            |> Chart.WithLayout layout
        
        let html = chart.GetHtml()
        File.WriteAllText(outputPath, html)
        printfn "DOM chart saved to %s" outputPath
        printfn "Date range: %O to %O (%d days)" dates[0] dates[dates.Length - 1] dates.Length

/// Calculate cumulative session VWAP from bar-level VWAP and volume
let private calculateSessionVwap (prices: Database.IntradayPriceRow array) : float array =
    let mutable cumVwapVolume = 0.0
    let mutable cumVolume = 0.0
    prices |> Array.map (fun p ->
        let barVwap = if p.vwap.HasValue then p.vwap.Value else (p.high + p.low + p.close) / 3.0
        cumVwapVolume <- cumVwapVolume + barVwap * p.volume
        cumVolume <- cumVolume + p.volume
        if cumVolume > 0.0 then cumVwapVolume / cumVolume else barVwap
    )

/// Generate an intraday candlestick chart with volume and VWAP, save as HTML
let generateIntradayCandlestickChart (prices: Database.IntradayPriceRow array) (ticker: string) (date: DateTime) (outputPath: string) (width: int) (height: int) : unit =
    let timestamps = prices |> Array.map (fun p -> p.timestamp)
    let volumes = prices |> Array.map (fun p -> p.volume)
    let barVwaps = prices |> Array.map (fun p -> 
        if p.vwap.HasValue then p.vwap.Value else (p.high + p.low + p.close) / 3.0)
    let sessionVwaps = calculateSessionVwap prices
    
    let candlestick =
        Candlestick(
            x = timestamps,
            ``open`` = (prices |> Array.map (fun p -> p.``open``)),
            high = (prices |> Array.map (fun p -> p.high)),
            low = (prices |> Array.map (fun p -> p.low)),
            close = (prices |> Array.map (fun p -> p.close)),
            name = ticker
        )
    
    let volume =
        Bar(
            x = timestamps,
            y = volumes,
            name = "Volume",
            marker = Marker(color = "rgba(100, 100, 200, 0.5)"),
            yaxis = "y2",
            hoverinfo = "y+x"
        )
    
    // Session VWAP line (cumulative)
    let sessionVwapTrace =
        Scatter(
            x = timestamps,
            y = sessionVwaps,
            name = "Session VWAP",
            mode = "lines",
            line = Line(color = "orange", width = 2.0),
            hoverinfo = "y+x+name"
        )
    
    // Bar VWAP as dots
    let barVwapTrace =
        Scatter(
            x = timestamps,
            y = barVwaps,
            name = "Bar VWAP",
            mode = "markers",
            marker = Marker(color = "rgba(128, 0, 128, 0.4)", size = 4),
            hoverinfo = "y+x+name"
        )
    
    let dateStr = date.ToString("yyyy-MM-dd")
    let layout =
        Layout(
            title = $"Intraday Chart - {ticker} ({dateStr})",
            xaxis = Xaxis(title = "Time"),
            yaxis = Yaxis(title = "Price", domain = [| 0.3; 1.0 |]),
            yaxis2 = Yaxis(title = "Volume", domain = [| 0.0; 0.25 |]),
            width = width,
            height = height,
            hovermode = "x unified"
        )
    
    let chart =
        [candlestick :> Trace; volume :> Trace; sessionVwapTrace :> Trace; barVwapTrace :> Trace]
        |> Chart.Plot
        |> Chart.WithLayout(layout)
    
    let html = chart.GetHtml()
    File.WriteAllText(outputPath, html)
    printfn "Chart saved to %s" outputPath

/// Generate an intraday chart for a ticker on a specific date
let generateIntradayChart (dbPath: string) (ticker: string) (date: DateTime) (timespan: string) (outputPath: string) (width: int) (height: int) : unit =
    if not (File.Exists dbPath) then
        failwithf "Database not found at %s" dbPath
    
    use connection = Database.openConnection dbPath
    
    let prices =
        match timespan with
        | "second" -> Database.getIntradaySecondByTickerDate connection ticker date
        | _ -> Database.getIntradayMinuteByTickerDate connection ticker date
    
    if prices.Length = 0 then
        printfn "No intraday data found for %s on %s" ticker (date.ToString("yyyy-MM-dd"))
    else
        printfn "Found %d %s bars for %s on %s" prices.Length timespan ticker (date.ToString("yyyy-MM-dd"))
        generateIntradayCandlestickChart prices ticker date outputPath width height
