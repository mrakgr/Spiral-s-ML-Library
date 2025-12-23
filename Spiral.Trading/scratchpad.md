# Overview

A few commands to get started for myself.

```
dotnet run --project Spiral.Trading.Console -- download-bulk -s 2003-09-10
dotnet run --project Spiral.Trading.Console -- download-splits -s 2003-09-10
dotnet run --project Spiral.Trading.Console -- ingest-data
```

Here is a command to plot the DOM.

```
dotnet run --project Spiral.Trading.Console -- plot-chart -t SPY
dotnet run --project Spiral.Trading.Console -- plot-dom -t SPY
```

Here is how to get trades data

```
dotnet run --project Spiral.Trading.Console -- download-trades -t LW -s 2025-12-19 --pretty
dotnet run --project Spiral.Trading.Console -- download-quotes -t LW -s 2025-12-19 --pretty
```