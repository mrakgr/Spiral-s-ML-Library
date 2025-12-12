using System;

namespace Spiral.Trading.Models
{
    public class Split
    {
        public int Id { get; set; }
        public string Ticker { get; set; } = string.Empty;
        public DateTime ExecutionDate { get; set; }
        public double SplitFrom { get; set; }
        public double SplitTo { get; set; }
        public double SplitRatio { get; set; }
    }
}
