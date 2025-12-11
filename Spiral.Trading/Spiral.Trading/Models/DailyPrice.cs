using System;

namespace Spiral.Trading.Models
{
    public class DailyPrice
    {
        public int Id { get; set; }
        public string Ticker { get; set; } = string.Empty;
        public DateTime Date { get; set; }
        public double Open { get; set; }
        public double High { get; set; }
        public double Low { get; set; }
        public double Close { get; set; }
        public long Volume { get; set; }
        public long WindowStart { get; set; }
        public int Transactions { get; set; }
    }
}
