# Congressional Trading Strategy System - Build Specification

## Data Sources (FREE, no API key required)

### 1. QuiverQuant Public Page Scraping
- URL: `https://www.quiverquant.com/congresstrading/`
- Returns 300 most recent congressional trades embedded in HTML
- Data format: Python list of lists
- Regex: `let recentTradesData = (\[.*?\])\s*;`
- Fields per trade (index-based):
  - [0] ticker: Stock symbol (e.g., "AAPL")
  - [1] asset_description: Full name (e.g., "Apple Inc. Common Stock")
  - [2] asset_type: "Stock", "ST", etc.
  - [3] trade_type: "Purchase", "Sale", "Sale (Full)"
  - [4] amount: Range string (e.g., "$50,001 - $100,000")
  - [5] representative: Name (e.g., "Nancy Pelosi")
  - [6] chamber: "Senate" or "House"
  - [7] party: "R" or "D"
  - [8] report_date: "YYYY-MM-DD HH:MM:SS"
  - [9] transaction_date: "YYYY-MM-DD HH:MM:SS"
  - [10] field11: usually "-"
  - [11] trade_id: unique ID string
  - [12] return_pct: float, return since trade
  - [13] full_name: official name
  - [14] photo_url: congress.gov image URL
  - [15] bioguide_id: official ID

### 2. Senate Stock Watcher (GitHub Archive)
- URL: `https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data/master/aggregate/all_transactions.json`
- Historical Senate trades (2015-2020, 8350 transactions, 6178 valid stock trades)
- Fields: transaction_date (MM/DD/YYYY), owner, ticker, asset_description, asset_type, type, amount, comment, senator, ptr_link

### 3. QuiverQuant Individual Stock Pages
- URL pattern: `https://www.quiverquant.com/congresstrading/stock/{TICKER}`
- Can get trade history for specific stocks

## Amount Ranges (for position sizing)
- "$1,001 - $15,000" → midpoint $8,000
- "$15,001 - $50,000" → midpoint $32,500
- "$50,001 - $100,000" → midpoint $75,000
- "$100,001 - $250,000" → midpoint $175,000
- "$250,001 - $500,000" → midpoint $375,000
- "$500,001 - $1,000,000" → midpoint $750,000
- "$1,000,001 - $5,000,000" → midpoint $3,000,000
- "$5,000,001 - $25,000,000" → midpoint $15,000,000
- "$25,000,001 - $50,000,000" → midpoint $37,500,000

## Signal Scoring System
Each trade is scored 0-100 based on:
1. **Trade Amount** (0-25 points): Higher amounts = stronger conviction
2. **Committee Relevance** (0-20 points): Members on relevant committees trading sector stocks
3. **Cluster Signal** (0-25 points): Multiple congress members buying same stock within 7 days
4. **Filing Speed** (0-15 points): Faster filing after transaction = more relevant
5. **Historical Success** (0-15 points): Track record of the specific member

## Risk Controls
- Max 10% of portfolio per position
- Max 30% sector concentration
- Stop loss at -8% per position
- Pause trading if portfolio drawdown > 15%
- Position sizing based on signal score: score 80+ = full position, 60-79 = half, below 60 = skip

## Committee Mapping (Key Committees → Sector Influence)
- Banking, Housing: Financials (XLF)
- Energy & Natural Resources: Energy (XLE)
- Commerce, Science: Technology (XLK), Communications (XLC)
- Health, Education: Healthcare (XLV)
- Armed Services: Defense/Aerospace (XLI, ITA)
- Agriculture: Consumer Staples (XLP)
- Finance Committee: Broad market, tax-sensitive sectors

## Alpaca Config
- API Key: PKMMCV2PAM45UCO7QNHNTL52SD
- API Secret: JxEct6xy512y8189skLkp7GekUKYnYUgjsPybtuPeBQ
- Base URL: https://paper-api.alpaca.markets
- Account: $100,000 paper trading
