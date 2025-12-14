#r "nuget: Microsoft.Data.Sqlite, 10.0.1"
#r "nuget: Dapper, 2.1.66"
#r "nuget: XPlot.Plotly, 4.1.0"
#r "../bin/Debug/net9.0/Spiral.Trading.dll"

Spiral.Trading.Plotting.generateDomChart 
    "../../../Python version/data/trading.db" 
    "dom_chart.html" 
    1200 
    600
