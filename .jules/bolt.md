## 2025-02-18 - [Fixing Broken Caching Pattern]
**Learning:** Code comments or documentation claiming a feature exists (read-through caching) can be misleading. Always verify the implementation. In this case, `StockDataFetcher` was writing to a cache but never reading from it, and used an insufficient key (ticker only vs ticker+period).
**Action:** When optimizing, first audit the existing "optimizations" to ensure they actually work. Fix broken patterns before adding new complexity.
