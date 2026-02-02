# Palette's Journal

## 2025-02-18 - Silent Truncation in Bulk Data
**Learning:** Users often assume systems process all provided data. Silently capping lists (e.g., first 50 items) for performance without notification breaks trust and causes confusion when expected results are missing.
**Action:** Always pair backend limits/truncations with clear frontend warnings or toasts.
