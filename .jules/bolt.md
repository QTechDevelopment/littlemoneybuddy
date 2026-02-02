## 2024-05-23 - Broken Caching Implementation
**Learning:** The `StockDataFetcher` class was writing to `self.cache` but never reading from it in `get_stock_data`. This resulted in 0% cache hit rate and repeated API calls/mock generation, defeating the purpose of the cache.
**Action:** Always verify that a cache is actually being READ from, not just written to. Verify cache hit rates in tests.

## 2024-05-23 - Batch Data Fetching
**Learning:** `yfinance` allows batch fetching via `download(tickers_list)`. This reduces N+1 HTTP requests to 1 request.
**Action:** Prefer `fetch_batch_data` pattern for pre-loading data when the set of keys (tickers) is known in advance (e.g., in a loop).

## 2024-05-24 - Missing Info Caching
**Learning:** `StockDataFetcher.get_stock_info` was fetching metadata individually for every ticker without caching, causing significant latency during portfolio analysis re-runs.
**Action:** Implemented `info_cache` to store and reuse stock metadata, reducing network calls for static data.
