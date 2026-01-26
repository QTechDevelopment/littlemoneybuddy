## 2024-05-23 - Broken Caching Implementation
**Learning:** The `StockDataFetcher` class was writing to `self.cache` but never reading from it in `get_stock_data`. This resulted in 0% cache hit rate and repeated API calls/mock generation, defeating the purpose of the cache.
**Action:** Always verify that a cache is actually being READ from, not just written to. Verify cache hit rates in tests.

## 2024-05-23 - Batch Data Fetching
**Learning:** `yfinance` allows batch fetching via `download(tickers_list)`. This reduces N+1 HTTP requests to 1 request.
**Action:** Prefer `fetch_batch_data` pattern for pre-loading data when the set of keys (tickers) is known in advance (e.g., in a loop).

## 2024-05-24 - YFinance Info Caching
**Learning:** `yfinance` does not support efficient batch retrieval for stock metadata (via `.info`). Accessing `.info` triggers an HTTP request per ticker, causing significant N+1 performance issues in loops.
**Action:** Implement explicit caching for singleton data fetches like `get_stock_info` when batch endpoints are unavailable, especially in stateful apps like Streamlit where objects persist in session state.
