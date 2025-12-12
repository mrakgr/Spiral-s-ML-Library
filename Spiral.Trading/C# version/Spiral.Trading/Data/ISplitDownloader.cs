using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Spiral.Trading.Models;

namespace Spiral.Trading.Data
{
    public interface ISplitDownloader
    {
        Task<List<Split>> DownloadAllSplitsAsync(DateTime startDate, DateTime? endDate = null);
    }
}
