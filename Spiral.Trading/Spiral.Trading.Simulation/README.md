# Spiral.Trading.Simulation

A market simulation framework for generating synthetic price data and backtesting trading strategies.

## Overview

This project generates realistic market data through a hierarchical episode structure:

1. **Day Sessions** - Morning, Mid, Close (sampled via MCMC)
2. **Trends** - Strong/Mid/Weak Up/Down trends and Consolidation (sampled via MCMC)
3. **Price Bars** - 1-second OHLC bars generated from trend parameters

## Installation

```bash
cd Spiral.Trading.Simulation
dotnet build
```

## CLI Commands

### Generate Day Structure

Generate session and trend episodes for a trading day:

```bash
dotnet run -- generate-day [-s seed] [-n runs] [-i iterations]
```

### Generate Price Bars

Generate 1-second price bars from episode structure:

```bash
dotnet run -- generate-prices [-s seed] [-i iterations] [-p start_price] [-o output.csv]
```

Export to CSV for analysis:

```bash
dotnet run -- generate-prices -o ../data/test_prices.csv
```

The CSV includes columns: `Time,Open,High,Low,Close,Session,Trend`

### Run MA Crossover Backtest

Backtest a moving average crossover strategy:

```bash
dotnet run -- backtest [-s seed] [-i iterations] [-f fast_period] [-l slow_period]
```

### Generate Order Book

Generate a simulated order book:

```bash
dotnet run -- order-book [-s seed]
```

## Architecture

### Modules

- **EpisodeMCMC.fs** - MCMC sampling for hierarchical episodes
- **PriceGeneration.fs** - Convert trends to price bars
- **MovingAverage.fs** - MA calculation and backtest logic
- **OrderBook.fs** - Order book simulation

### Episode Structure

```
Day (390 minutes)
├── Morning (~60 min) - Higher probability of strong trends
├── Mid (~270 min) - Consolidation-heavy
└── Close (~60 min) - Higher probability of strong trends
    └── Trends
        ├── StrongUptrend (short duration, high drift)
        ├── MidUptrend
        ├── WeakUptrend
        ├── Consolidation (no drift)
        ├── WeakDowntrend
        ├── MidDowntrend
        └── StrongDowntrend (short duration, high drift)
```

### MCMC Sampling

Episodes are sampled using Metropolis-Hastings with:
- **Duration transfers** between adjacent episodes
- **Label changes** for trend types
- **Log-normal distributions** for duration likelihoods

## Example Output

```
$ dotnet run -- backtest

Price Summary:
  Bars: 23391 (389.9 minutes)
  Open:  100.0000
  Close: 106.1238
  High:  106.6105
  Low:   97.6112
  Return: 6.12%

MA Crossover Backtest (fast=10, slow=30):
  Trades: 942
  Total PnL: 1.1745
  Win Rate: 37.8%
```
