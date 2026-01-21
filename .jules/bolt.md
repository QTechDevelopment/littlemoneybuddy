# Bolt's Journal

## 2024-05-22 - [StockDataFetcher Caching]
**Learning:** `yfinance` calls can be slow or fail, especially in restricted environments. The application was re-fetching data on every call because caching was write-only (cache was populated but never checked).
**Action:** Always verify that a caching mechanism actually *reads* from the cache before fetching data.
