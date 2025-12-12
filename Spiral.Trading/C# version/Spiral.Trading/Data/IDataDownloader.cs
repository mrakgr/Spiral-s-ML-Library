using System;
using System.Threading.Tasks;

namespace Spiral.Trading.Data
{
    public interface IDataDownloader
    {
        Task DownloadDailyAggregatesAsync(DateTime startDate, DateTime endDate, string outputDirectory, int maxDegreeOfParallelism = 20);
    }
}
