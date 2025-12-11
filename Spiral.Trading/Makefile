PROJECT_PATH = Spiral.Trading.Console/Spiral.Trading.Console.fsproj

.PHONY: all build ingest splits bulk clean help

all: build

build:
	dotnet build $(PROJECT_PATH)

ingest:
	dotnet run --project $(PROJECT_PATH) -- --ingest-data

splits:
	dotnet run --project $(PROJECT_PATH) -- --download-splits

bulk:
	dotnet run --project $(PROJECT_PATH) -- --download-bulk

plot:
	dotnet run --project $(PROJECT_PATH) -- --plot --ticker $(or $(TICKER),NVDA)

all: build bulk ingest splits

clean:
	dotnet clean $(PROJECT_PATH)

help:
	@echo "Available commands:"
	@echo "  make build    - Build the solution"
	@echo "  make ingest   - Run data ingestion (daily aggregates -> DB)"
	@echo "  make splits   - Download stock splits for tickers in DB"
	@echo "  make bulk     - Run bulk download (S3 -> disk)"
	@echo "  make all      - Run all tasks (build, bulk, ingest, splits)"
	@echo "  make plot     - Generate chart for ticker (usage: make plot TICKER=NVDA)"
	@echo "  make clean    - Clean build artifacts"
