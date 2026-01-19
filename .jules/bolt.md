## 2024-05-22 - Missing Cache Read in Data Fetcher
**Learning:** Implemented caching (writing to dictionary) is useless without a corresponding read check. Always verify both read and write paths when implementing caching.
**Action:** When reviewing caching logic, explicitly check for `if key in cache: return cache[key]` at the start of the function.

## 2024-05-22 - Caching Fallbacks
**Learning:** Caching failure states (like mock data fallbacks) is as important as caching success states to prevent repeated expensive failures (e.g., network timeouts).
**Action:** Cache the result of fallbacks or negative lookups to protect the system from repeated external failures.
