import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from stock_data import StockDataFetcher

class TestStockDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = StockDataFetcher()

    @patch('stock_data.yf')
    def test_fetch_batch_data_success(self, mock_yf):
        # Mock the download response
        # Create a MultiIndex DataFrame to simulate batch download
        tickers = ['AAPL', 'MSFT']
        dates = pd.date_range(start='2023-01-01', periods=5)

        # Create mock data
        data_aapl = pd.DataFrame({'Close': [150]*5}, index=dates)
        data_msft = pd.DataFrame({'Close': [300]*5}, index=dates)

        # Construct MultiIndex columns
        # (Ticker, Price)
        columns = pd.MultiIndex.from_product([tickers, ['Close']], names=['Ticker', 'Price'])
        # Concatenate data
        data = pd.concat([data_aapl, data_msft], axis=1)
        data.columns = columns

        # Mock download return
        mock_yf.download.return_value = data

        self.fetcher.fetch_batch_data(tickers, period="3mo")

        # Check if cache is populated
        self.assertIn(('AAPL', '3mo'), self.fetcher.cache)
        self.assertIn(('MSFT', '3mo'), self.fetcher.cache)

        # Check cache content
        # Note: data['AAPL'] returns a DataFrame with 'Close' column but not MultiIndex columns
        pd.testing.assert_frame_equal(self.fetcher.cache[('AAPL', '3mo')], data['AAPL'])
        pd.testing.assert_frame_equal(self.fetcher.cache[('MSFT', '3mo')], data['MSFT'])

    @patch('stock_data.yf')
    def test_fetch_batch_data_single_ticker(self, mock_yf):
        # Mock response for single ticker (which yf might return as single level or multi level depending on input)
        # Here we simulate the case where we request one ticker via batch
        tickers = ['AAPL']
        dates = pd.date_range(start='2023-01-01', periods=5)
        data = pd.DataFrame({'Close': [150]*5}, index=dates)

        mock_yf.download.return_value = data

        self.fetcher.fetch_batch_data(tickers, period="3mo")

        self.assertIn(('AAPL', '3mo'), self.fetcher.cache)
        pd.testing.assert_frame_equal(self.fetcher.cache[('AAPL', '3mo')], data)

    @patch('stock_data.yf')
    def test_get_stock_data_uses_cache(self, mock_yf):
        # Pre-populate cache
        dates = pd.date_range(start='2023-01-01', periods=5)
        data = pd.DataFrame({'Close': [150]*5}, index=dates)
        self.fetcher.cache[('AAPL', '3mo')] = data

        result = self.fetcher.get_stock_data('AAPL', '3mo')

        # Should return cached data and NOT call yf.Ticker
        pd.testing.assert_frame_equal(result, data)
        mock_yf.Ticker.assert_not_called()

    @patch('stock_data.yf')
    def test_get_stock_info_uses_cache(self, mock_yf):
        # 1. First call - should hit API
        mock_ticker = MagicMock()
        mock_ticker.info = {
            'symbol': 'AAPL',
            'longName': 'Apple Inc.',
            'sector': 'Technology'
        }
        mock_yf.Ticker.return_value = mock_ticker

        # Verify cache is empty initially
        self.assertNotIn('AAPL', self.fetcher.info_cache)

        result1 = self.fetcher.get_stock_info('AAPL')

        # Verify result and cache population
        self.assertEqual(result1['symbol'], 'AAPL')
        self.assertIn('AAPL', self.fetcher.info_cache)
        self.assertEqual(self.fetcher.info_cache['AAPL'], result1)

        # Verify API called once
        mock_yf.Ticker.assert_called_once_with('AAPL')

        # 2. Second call - should use cache
        # Reset mock to ensure it's not called again
        mock_yf.Ticker.reset_mock()

        result2 = self.fetcher.get_stock_info('AAPL')

        # Verify result is same
        self.assertEqual(result2, result1)

        # Verify API NOT called
        mock_yf.Ticker.assert_not_called()

if __name__ == '__main__':
    unittest.main()
