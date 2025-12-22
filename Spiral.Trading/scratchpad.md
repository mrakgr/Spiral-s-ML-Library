# Overview

A few commands to get started for myself.

```
dotnet run --project Spiral.Trading.Console -- download-bulk -s 1/1/2015
dotnet run --project Spiral.Trading.Console -- download-splits -s 1/1/2015
dotnet run --project Spiral.Trading.Console -- ingest-data
```

Here is a command to plot the DOM with IWM.

```
dotnet run --project Spiral.Trading.Console -- plot-chart -t IWM
dotnet run --project Spiral.Trading.Console -- plot-dom -t IWM
```

Here is how to get trades data

```
dotnet run --project Spiral.Trading.Console -- download-trades -t LW -s 2025-12-19 --pretty
```