using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using XPlot.Plotly;
using Spiral.Trading.Models;

namespace Spiral.Trading.Plotting
{
    public class ChartGenerator
    {
        public static void GenerateCandlestickChart(List<DailyPrice> prices, string ticker, string outputPath)
        {
            var chart = Chart.Plot(
                new Candlestick
                {
                    x = prices.Select(p => p.Date.ToString("yyyy-MM-dd")).ToArray(),
                    open = prices.Select(p => p.Open).ToArray(),
                    high = prices.Select(p => p.High).ToArray(),
                    low = prices.Select(p => p.Low).ToArray(),
                    close = prices.Select(p => p.Close).ToArray(),
                    name = ticker
                }
            );

            chart.WithLayout(new Layout.Layout
            {
                title = $"Daily Price Chart - {ticker}",
                xaxis = new Xaxis { title = "Date" },
                yaxis = new Yaxis { title = "Price" }
            });

            // XPlot.Plotly usually generates a full HTML page or a div.
            // SaveHtml wraps it in a basic template.
            var html = chart.GetHtml();

            // We can wrap it in a complete HTML document if GetHtml() returns just the div/script.
            // XPlot usually returns a full string with GetHtml().
            // Wait, Chart.Save(path) is available but might rely on internet for plotly.js CDN.
            // Let's use File.WriteAllText for control.

            // Actually, let's use the builtin Save if possible, but XPlot sometimes acts up with paths.
            // Let's try explicit HTML construction for maximum reliability if needed, 
            // but for now relying on library method.

            // Note: XPlot.Plotly 4.1.0's Chart.GetHtml() returns the full HTML string.
            File.WriteAllText(outputPath, html);
            Console.WriteLine($"Chart saved to {outputPath}");
        }
    }
}
