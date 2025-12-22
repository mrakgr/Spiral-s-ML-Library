# Trade Conditions Reference

This document explains the trade condition codes that appear in US equity trades data. These conditions are reported by exchanges to the SIP (Securities Information Processor) and indicate special circumstances about how a trade was executed.

## Understanding Trade Conditions

Each trade can have zero or more condition codes in the `conditions` array. These codes affect:
- Whether the trade updates the official high/low prices
- Whether the trade updates the official open/close prices
- Whether the trade counts toward volume calculations

## Sale Conditions

These are the most common conditions you'll encounter in trades data.

### Regular Trading

| ID | Name                          | Description                                                                                                                          | Hi/Lo | Op/Cl | Volume |
|---:|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
|  1 | Acquisition                   | Trade related to an acquisition or merger. Treated as a regular sale for calculation purposes.                                       |   Y   |   Y   |   Y    |
|  3 | Automatic Execution           | Trade was executed automatically by the exchange's matching engine without manual intervention. Standard electronic execution.       |   Y   |   Y   |   Y    |
|  4 | Bunched Trade                 | Multiple orders from the same firm were grouped together and executed as a single trade.                                             |   Y   |   Y   |   Y    |
|  5 | Bunched Sold Trade            | A bunched trade that was reported late (out of sequence). Updates volume but not open/close.                                         |   Y   |   N   |   Y    |
|  9 | Cross Trade                   | A trade where the same broker represents both the buyer and seller. Must be executed at or between the NBBO.                         |   Y   |   Y   |   Y    |
| 11 | Distribution                  | A large block trade distributed to multiple buyers, often from an institutional seller.                                              |   Y   |   Y   |   Y    |
| 23 | Rule 155 Trade (AMEX)         | Trade executed under AMEX Rule 155, which allows specialists to facilitate large orders.                                             |   Y   |   Y   |   Y    |
| 24 | Rule 127 (NYSE Only)          | Trade executed under NYSE Rule 127, allowing floor brokers to cross orders outside the quote.                                        |   Y   |   Y   |   Y    |
| 27 | Stopped Stock (Regular Trade) | A trade where a market maker guaranteed a price to a customer and later executed at that price or better.                            |   Y   |   Y   |   Y    |
| 30 | Sold Last                     | Trade reported late but still updates last sale price.                                                                               |   Y   |   Y   |   Y    |
| 31 | Sold Last and Stopped Stock   | Combination of stopped stock and late reporting.                                                                                     |   Y   |   Y   |   Y    |
| 34 | Split Trade                   | A single order that was split and executed in multiple parts.                                                                        |   Y   |   Y   |   Y    |
| 36 | Yellow Flag Regular Trade     | Trade flagged for regulatory review but otherwise treated normally.                                                                  |   Y   |   Y   |   Y    |

### Opening and Closing

| ID | Name                          | Description                                                                                                  | Hi/Lo | Op/Cl | Volume |
|---:|-------------------------------|--------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
|  8 | Closing Prints                | Trades executed in the closing auction. These set the official closing price.                                |   Y   |   Y   |   Y    |
| 15 | Market Center Official Close  | The official closing price for a specific exchange. A price marker, not a trade.                             |   N   |   N   |   N    |
| 16 | Market Center Official Open   | The official opening price for a specific exchange. A price marker, not a trade.                             |   N   |   N   |   N    |
| 17 | Market Center Opening Trade   | First trade of the day at a specific exchange, typically from the opening auction.                           |   Y   |   Y   |   Y    |
| 18 | Market Center Reopening Trade | First trade after a trading halt at a specific exchange.                                                     |   Y   |   Y   |   Y    |
| 19 | Market Center Closing Trade   | Last regular trade at a specific exchange before the close.                                                  |   Y   |   Y   |   Y    |
| 25 | Opening Prints                | Trades executed in the opening auction.                                                                      |   Y   |   Y   |   Y    |
| 28 | Re-Opening Prints             | Trades from the reopening auction after a halt.                                                              |   Y   |   Y   |   Y    |
| 55 | Opening Reopening Trade Detail| Detailed information about opening or reopening auction trades.                                              |   Y   |   Y   |   Y    |

### Extended Hours Trading

| ID | Name                                  | Description                                                                                                                                              | Hi/Lo | Op/Cl | Volume |
|---:|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
| 12 | Form T/Extended Hours                 | Trade executed outside regular market hours (pre-market 4:00-9:30 AM or after-hours 4:00-8:00 PM ET). Named after the SEC form used to report these.    |   N   |   N   |   Y    |
| 13 | Extended Hours (Sold Out Of Sequence) | Extended hours trade that was reported late or out of sequence.                                                                                          |   N   |   N   |   Y    |

### Special Pricing

| ID | Name                 | Description                                                                                                                                                                       | Hi/Lo | Op/Cl | Volume |
|---:|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
|  2 | Average Price Trade  | Trade executed at a VWAP or other averaged price. Common for institutional orders. Excluded from hi/lo and open/close because the price doesn't reflect a point-in-time market. |   N   |   N   |   Y    |
|  6 | CAP Election         | Trade related to a Conditional Auction Process, used in certain exchange mechanisms.                                                                                              |   Y   |   Y   |   Y    |
| 10 | Derivatively Priced  | Trade price was derived from another instrument (e.g., an ETF priced based on its underlying basket).                                                                             |   Y   |   N   |   Y    |
| 21 | Price Variation Trade| Trade executed at a price that varies significantly from the current market. Often used for negotiated trades or error corrections.                                               |   N   |   N   |   Y    |
| 22 | Prior Reference Price| Trade executed at a previously established reference price rather than the current market price.                                                                                  |   Y   |   N   |   Y    |

### Settlement Variations

| ID | Name      | Description                                                                                                                                    | Hi/Lo | Op/Cl | Volume |
|---:|-----------|------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
|  7 | Cash Sale | Trade that settles same-day (T+0) instead of standard T+1. Excluded from hi/lo and open/close because price may reflect settlement premium.   |   N   |   N   |   Y    |
| 20 | Next Day  | Trade that settles on the next business day. Excluded from hi/lo and open/close.                                                               |   N   |   N   |   Y    |
| 29 | Seller    | Trade with special seller's terms, typically extended settlement.                                                                              |   N   |   N   |   Y    |

### Out of Sequence / Late Reports

| ID | Name                                    | Description                                                                                                           | Hi/Lo | Op/Cl | Volume |
|---:|-----------------------------------------|-----------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
| 32 | Sold (Out Of Sequence)                  | Trade reported late, after subsequent trades have already been reported. Updates hi/lo but not open/close.            |   Y   |   N   |   Y    |
| 33 | Sold (Out of Sequence) and Stopped Stock| Combination of out-of-sequence reporting and stopped stock.                                                           |   Y   |   N   |   Y    |

### Odd Lots

| ID | Name           | Description                                                                                                                                                                  | Hi/Lo | Op/Cl | Volume |
|---:|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
| 37 | Odd Lot Trade  | Trade for fewer than 100 shares. Historically excluded from the consolidated tape entirely. Now reported but excluded from hi/lo and open/close. Still counts toward volume.|   N   |   N   |   Y    |

### Contingent and Complex Trades

| ID | Name                           | Description                                                                                                                                        | Hi/Lo | Op/Cl | Volume |
|---:|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
| 35 | Stock Option                   | Trade executed as part of a stock-option combination order (e.g., buy-write, covered call). Stock leg may be priced relative to the option.       |   Y   |   Y   |   Y    |
| 52 | Contingent Trade               | Trade contingent on another event or transaction. Excluded from price calculations because the price may not reflect true market value.           |   N   |   N   |   Y    |
| 53 | Qualified Contingent Trade     | A multi-component trade where the stock leg is contingent on other legs (often derivatives). Exempt from certain trade-through rules.             |   N   |   N   |   Y    |

### Corrections

| ID | Name                        | Description                                                                                                    | Hi/Lo | Op/Cl | Volume |
|---:|-----------------------------|----------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
| 38 | Corrected Consolidated Close| A correction to the official closing price. Updates hi/lo and open/close but not volume (already counted).    |   Y   |   Y   |   N    |

## Trade-Through Exemptions

| ID | Name             | Description                                                                                                                                                                                                  | Hi/Lo | Op/Cl | Volume |
|---:|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----:|:------:|
| 41 | Trade Thru Exempt| Trade is exempt from Regulation NMS trade-through rules. The trade was allowed to execute at a price that might appear worse than quotes on other exchanges. Common for ISOs, benchmark trades, self-help. |   Y   |   Y   |   Y    |

## Short Sale Restriction Indicators

These indicate the SSR (Short Sale Restriction) status for a security. They typically appear in reference data rather than on individual trades.

| ID | Name                                 | Description                                                                                        |
|---:|--------------------------------------|----------------------------------------------------------------------------------------------------|
| 57 | Short Sale Restriction Activated     | SSR was triggered today (stock dropped 10%+ from previous close). Short sales restricted to uptick.|
| 58 | Short Sale Restriction Continued     | SSR remains in effect from the previous trading day.                                               |
| 59 | Short Sale Restriction Deactivated   | SSR has been lifted.                                                                               |
| 60 | Short Sale Restriction In Effect     | SSR is currently active for this security.                                                         |

## Financial Status Indicators

These flag securities with financial or regulatory compliance issues. They appear in reference data to warn traders about the issuer's status.

| ID | Name                                         | Description                                                            |
|---:|----------------------------------------------|------------------------------------------------------------------------|
| 62 | Financial Status - Bankrupt                  | The issuer has filed for bankruptcy.                                   |
| 63 | Financial Status - Deficient                 | The issuer is deficient in meeting exchange listing requirements.      |
| 64 | Financial Status - Delinquent                | The issuer is delinquent in required SEC filings.                      |
| 65 | Financial Status - Bankrupt and Deficient    | Both bankrupt and deficient.                                           |
| 66 | Financial Status - Bankrupt and Delinquent   | Both bankrupt and delinquent in filings.                               |
| 67 | Financial Status - Deficient and Delinquent  | Both deficient in listing requirements and delinquent in filings.      |
| 68 | Financial Status - Deficient, Delinquent, Bankrupt | All three issues.                                                |
| 69 | Financial Status - Liquidation               | The company is being liquidated.                                       |
| 70 | Financial Status - Creations Suspended       | For ETFs: the creation of new shares is suspended.                     |
| 71 | Financial Status - Redemptions Suspended     | For ETFs: the redemption of shares is suspended.                       |

## Intermarket Sweep Orders (ISO)

When you see condition 14 (Intermarket Sweep), it indicates the trade was part of an ISO. This is a special order type where:

1. The trader wants to execute immediately at a specific price
2. They simultaneously send orders to all exchanges displaying better prices
3. This "sweeps" the book across multiple venues

ISOs are exempt from trade-through rules because the trader is responsible for clearing out better-priced quotes. They're commonly used by algorithms and institutional traders for fast execution.

## Common Condition Combinations

Trades often have multiple conditions. Common combinations include:

- `[14, 41]` - Intermarket sweep that's trade-through exempt
- `[37, 12]` - Odd lot trade in extended hours
- `[37, 41]` - Odd lot that's trade-through exempt
- `[8, 14]` - Closing print from an intermarket sweep

## Filtering Trades for Analysis

Depending on your analysis, you may want to filter trades:

**For price discovery analysis (what's the "real" price):**
- Exclude: 2, 7, 12, 13, 20, 21, 29, 37, 52, 53
- These are either averaged prices, special settlement, extended hours, odd lots, or contingent trades

**For volume analysis:**
- Include almost everything (most conditions update volume)
- Exclude: 15, 16, 38 (these are price markers or corrections, not actual trades)

**For VWAP calculation:**
- Typically include all trades that update volume
- May want to exclude extended hours (12, 13) depending on your definition

**For official OHLC:**
- Only include trades where Updates Op/Cl = Y
- This matches how the consolidated tape calculates official prices
