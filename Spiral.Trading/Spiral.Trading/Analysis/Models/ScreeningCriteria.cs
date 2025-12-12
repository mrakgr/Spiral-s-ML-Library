namespace Spiral.Trading.Analysis.Models;

/// <summary>
/// Configuration for stock screening criteria
/// </summary>
public class ScreeningCriteria
{
    public double MinRelativeVolume { get; set; } = 3.0;
    public double MinPriceChangePercent { get; set; } = 5.0;
    public int MinimumDaysInMonth { get; set; } = 10;
    public bool RequireBothCriteria { get; set; } = false; // AND vs OR logic
}
