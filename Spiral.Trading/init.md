# Overview

A few commands to get started for myself.

```
dotnet run --project Spiral.Trading.Console -- download-bulk -s 1/1/2016
dotnet run --project Spiral.Trading.Console -- download-splits -s 1/1/2016
dotnet run --project Spiral.Trading.Console -- ingest-data
```

Here is a command to plot the DOM with QQQ.

```
dotnet run --project Spiral.Trading.Console -- plot-chart -t NVDA
dotnet run --project Spiral.Trading.Console -- plot-dom -t SPY
```