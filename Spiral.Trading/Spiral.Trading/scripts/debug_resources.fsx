#r "nuget: Microsoft.Data.Sqlite, 10.0.1"
#r "nuget: Dapper, 2.1.66"
#r "nuget: XPlot.Plotly, 4.1.0"
#r "../bin/Debug/net9.0/Spiral.Trading.dll"

open System.Reflection

let assembly = Assembly.LoadFrom("../bin/Debug/net9.0/Spiral.Trading.dll")
let resources = assembly.GetManifestResourceNames()

printfn "Embedded resources:"
for r in resources do
    printfn "  %s" r
