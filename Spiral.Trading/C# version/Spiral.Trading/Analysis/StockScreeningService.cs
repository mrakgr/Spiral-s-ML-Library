using System.Globalization;
using CsvHelper;
using Microsoft.EntityFrameworkCore;
using Spiral.Trading.Analysis.Models;
using Spiral.Trading.Models;
using Spiral.Trading.Storage;

namespace Spiral.Trading.Analysis;

/// <summary>
/// Service for screening stocks based on relative volume and price movement
/// </summary>
public class StockScreeningService
{
    private readonly string _dbPath;

    public StockScreeningService(string dbPath)
    {
        _dbPath = dbPath;
    }

    /// <summary>
    /// Screen for "Stocks In Play" based on relative volume and price movement.
    /// Calculates monthly average volume EXCLUDING the current date for each day.
    /// </summary>
    public async Task<List<StockInPlayResult>> ScreenStocksInPlayAsync(
        int year,
        int month,
        ScreeningCriteria? criteria = null)
    {
        criteria ??= new ScreeningCriteria();

        var startDate = new DateTime(year, month, 1);
        var endDate = startDate.AddMonths(1);

        using var context = new TradingDbContext(_dbPath);

        // Load all prices for the month - AsNoTracking for performance
        var monthlyPrices = await context.DailyPrices
            .AsNoTracking()
            .Where(p => p.Date >= startDate && p.Date < endDate)
            .ToListAsync();

        Console.WriteLine($"Loaded {monthlyPrices.Count:N0} price records for {year}-{month:D2}");

        // Group by ticker and analyze each stock
        var results = monthlyPrices
            .GroupBy(p => p.Ticker)
            .Where(g => g.Count() >= criteria.MinimumDaysInMonth)
            .SelectMany(group => AnalyzeTickerDays(group.Key, group.ToList(), criteria))
            .OrderByDescending(r => r.RelativeVolume)
            .ToList();

        return results;
    }

    /// <summary>
    /// Analyze each day for a ticker to find "in play" days.
    /// For each day, calculates monthly average volume EXCLUDING that day.
    /// </summary>
    private List<StockInPlayResult> AnalyzeTickerDays(
        string ticker,
        List<DailyPrice> prices,
        ScreeningCriteria criteria)
    {
        var results = new List<StockInPlayResult>();
        var sortedPrices = 
            prices.OrderBy(p => p.Date)
                .ToList();

        foreach (var currentDay in sortedPrices)
        {
            // Calculate monthly average volume EXCLUDING current day
            var otherDays = sortedPrices.Where(p => p.Date != currentDay.Date).ToList();

            if (otherDays.Count == 0) continue;

            var avgVolume = otherDays.Average(p => (double)p.Volume);
            var relativeVolume = avgVolume > 0 ? currentDay.Volume / avgVolume : 0;

            // Calculate daily price change percentage
            var dailyChangePercent = currentDay.Open > 0
                ? (currentDay.Close - currentDay.Open) / currentDay.Open * 100
                : 0;

            // Determine if stock meets criteria
            bool meetsVolumeCriteria = relativeVolume >= criteria.MinRelativeVolume;
            bool meetsPriceCriteria = Math.Abs(dailyChangePercent) >= criteria.MinPriceChangePercent;

            bool qualifies = criteria.RequireBothCriteria
                ? (meetsVolumeCriteria && meetsPriceCriteria)
                : (meetsVolumeCriteria || meetsPriceCriteria);

            if (qualifies)
            {
                string reason = meetsVolumeCriteria && meetsPriceCriteria ? "Both"
                    : meetsVolumeCriteria ? "High Volume"
                    : "Price Move";

                results.Add(new StockInPlayResult
                {
                    Ticker = ticker,
                    Date = currentDay.Date,
                    Open = currentDay.Open,
                    Close = currentDay.Close,
                    Volume = currentDay.Volume,
                    DailyChangePercent = dailyChangePercent,
                    MonthlyAvgVolume = avgVolume,
                    RelativeVolume = relativeVolume,
                    Reason = reason
                });
            }
        }

        return results;
    }

    /// <summary>
    /// Export screening results to CSV
    /// </summary>
    public void ExportToCsv(List<StockInPlayResult> results, string outputPath)
    {
        using var writer = new StreamWriter(outputPath);
        using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);

        csv.WriteRecords(results);
        Console.WriteLine($"Exported {results.Count} results to {outputPath}");
    }

    /// <summary>
    /// Print results to console in formatted table
    /// </summary>
    public void PrintResults(List<StockInPlayResult> results, int maxResults = 50)
    {
        Console.WriteLine($"\nFound {results.Count} 'Stocks In Play' instances\n");

        if (results.Count == 0)
        {
            Console.WriteLine("No stocks met the screening criteria.");
            return;
        }

        Console.WriteLine($"{"Date",-12} {"Ticker",-8} {"Open",10} {"Close",10} {"Change %",10} {"Volume",12} {"Avg Vol",12} {"Rel Vol",8} {"Reason",-12}");
        Console.WriteLine(new string('-', 110));

        var topResults = results.Take(maxResults);

        foreach (var r in topResults)
        {
            Console.WriteLine(
                $"{r.Date:yyyy-MM-dd}   {r.Ticker,-8} {r.Open,10:F2} {r.Close,10:F2} " +
                $"{r.DailyChangePercent,9:F2}% {r.Volume,12:N0} {r.MonthlyAvgVolume,12:N0} " +
                $"{r.RelativeVolume,7:F2}x {r.Reason,-12}");
        }

        if (results.Count > maxResults)
        {
            Console.WriteLine($"\n... and {results.Count - maxResults} more results");
        }
    }
}
