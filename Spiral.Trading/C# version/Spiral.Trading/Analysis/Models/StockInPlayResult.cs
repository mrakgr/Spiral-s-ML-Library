namespace Spiral.Trading.Analysis.Models;

/// <summary>
/// Represents a stock that meets "In Play" criteria on a specific date
/// </summary>
public class StockInPlayResult
{
    public string Ticker { get; set; } = string.Empty;
    public DateTime Date { get; set; }
    public double Open { get; set; }
    public double Close { get; set; }
    public long Volume { get; set; }
    public double DailyChangePercent { get; set; }
    public double MonthlyAvgVolume { get; set; }
    public double RelativeVolume { get; set; }
    public string Reason { get; set; } = string.Empty; // "High Volume", "Price Move", "Both"
}
