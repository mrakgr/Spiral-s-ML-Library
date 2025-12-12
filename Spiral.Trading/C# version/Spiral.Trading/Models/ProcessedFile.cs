using System;

namespace Spiral.Trading.Models
{
    public class ProcessedFile
    {
        public int Id { get; set; }
        public string FileName { get; set; } = string.Empty;
        public DateTime IngestedAt { get; set; }
    }
}
