## 2024-05-22 - Broken Caching Pattern
**Learning:** Found a "write-only" cache in `StockDataFetcher`. The code was populating `self.cache` but never checking it before making expensive API calls. This is a critical pattern to watch for: code that *looks* like it has caching (because it assigns to a cache variable) but is functionally equivalent to no caching.
**Action:** When auditing caching systems, always grep for both read AND write operations to the cache variable.
