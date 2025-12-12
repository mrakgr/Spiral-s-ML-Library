using Microsoft.EntityFrameworkCore;
using Spiral.Trading.Models;

namespace Spiral.Trading.Storage
{
    public class TradingDbContext(string dbPath = "data/trading.db") : DbContext
    {
        public DbSet<DailyPrice> DailyPrices { get; set; } = null!;
        public DbSet<ProcessedFile> ProcessedFiles { get; set; } = null!;
        public DbSet<Split> Splits { get; set; } = null!;

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlite($"Data Source={dbPath}");
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            // DailyPrice Configuration
            modelBuilder.Entity<DailyPrice>()
                .HasKey(d => d.Id); // We need to add ID to the model or configure composite key

            modelBuilder.Entity<DailyPrice>()
                .HasIndex(d => new { d.Ticker, d.Date })
                .IsUnique();

            modelBuilder.Entity<DailyPrice>()
                .HasIndex(d => d.Date);

            modelBuilder.Entity<DailyPrice>()
                .HasIndex(d => d.Ticker);

            // Split Configuration
            modelBuilder.Entity<Split>()
                .HasKey(s => s.Id);

            modelBuilder.Entity<Split>()
                .HasIndex(s => new { s.Ticker, s.ExecutionDate })
                .IsUnique();

            modelBuilder.Entity<Split>()
                .HasIndex(s => s.Ticker);

            modelBuilder.Entity<Split>()
                .HasIndex(s => s.ExecutionDate);
        }
    }
}
