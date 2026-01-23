## 2024-05-23 - StockDataFetcher Write-Only Cache
**Learning:** The `StockDataFetcher` class was writing to a cache but never reading from it, resulting in redundant API calls. Additionally, the cache key was insufficient (missing `period`), which would have caused collisions if it were used.
**Action:** When auditing caching implementations, always verify both the read and write paths and ensure cache keys cover all varying parameters.
