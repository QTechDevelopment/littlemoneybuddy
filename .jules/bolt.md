## 2026-01-24 - yfinance Batch Fetching Optimization
**Learning:** Sequential `yfinance` calls for multiple tickers are extremely slow due to HTTP overhead and rate limiting (429s). `yf.download(group_by='ticker')` returns a MultiIndex DataFrame that can be parsed to populate a cache efficiently, reducing N requests to 1.
**Action:** When working with financial data APIs, always prefer batch/bulk endpoints. Ensure to handle MultiIndex vs Flat DataFrame return types when parsing bulk responses.

## 2026-01-24 - Cache Key Specificity
**Learning:** Caching by `ticker` alone is insufficient when the data depends on time `period`. This can lead to serving stale or incorrect data ranges.
**Action:** Always include all variable parameters (e.g., `(ticker, period)`) in cache keys to ensure data correctness.
