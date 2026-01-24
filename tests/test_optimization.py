
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_data import StockDataFetcher

class TestStockDataOptimization(unittest.TestCase):
    def setUp(self):
        self.fetcher = StockDataFetcher()

    @patch('stock_data.yf')
    def test_fetch_batch_data_populates_cache(self, mock_yf):
        # Setup mock return value for download
        # We need to simulate a MultiIndex DataFrame return
        tickers = ['AAPL', 'MSFT']
        period = '1mo'

        # Create a mock DataFrame with MultiIndex columns
        # Level 0: Ticker, Level 1: Price
        columns = pd.MultiIndex.from_product([tickers, ['Open', 'Close', 'High', 'Low', 'Volume']], names=['Ticker', 'Price'])
        data = pd.DataFrame(index=range(5), columns=columns)
        # Fill with some dummy data
        data.loc[:, ('AAPL', 'Close')] = 150.0
        data.loc[:, ('MSFT', 'Close')] = 300.0

        mock_yf.download.return_value = data

        # Call batch fetch
        self.fetcher.fetch_batch_data(tickers, period)

        # Verify download was called correctly
        mock_yf.download.assert_called_once()
        args, kwargs = mock_yf.download.call_args
        self.assertEqual(kwargs['group_by'], 'ticker')
        self.assertIn('AAPL', kwargs['tickers'] if 'tickers' in kwargs else args[0])

        # Verify cache population
        self.assertIn(('AAPL', period), self.fetcher.cache)
        self.assertIn(('MSFT', period), self.fetcher.cache)

        # Verify content
        aapl_data = self.fetcher.cache[('AAPL', period)]
        self.assertFalse(aapl_data.empty)
        # Check if it has Close column (it might be flat or not depending on how we constructed mock)
        # Our implementation slices: data['AAPL'] which returns DataFrame with Price columns
        self.assertIn('Close', aapl_data.columns)

    @patch('stock_data.yf')
    def test_get_stock_data_uses_cache(self, mock_yf):
        ticker = 'GOOGL'
        period = '3mo'

        # Pre-populate cache
        mock_df = pd.DataFrame({'Close': [100, 101]})
        self.fetcher.cache[(ticker, period)] = mock_df

        # Call get_stock_data
        result = self.fetcher.get_stock_data(ticker, period)

        # Verify result is from cache
        pd.testing.assert_frame_equal(result, mock_df)

        # Verify yf.Ticker was NOT called
        mock_yf.Ticker.assert_not_called()

    @patch('stock_data.yf')
    def test_get_stock_info_caching(self, mock_yf):
        ticker = 'NVDA'
        mock_info = {'symbol': 'NVDA', 'longName': 'NVIDIA Corp', 'sector': 'Technology'}

        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_yf.Ticker.return_value = mock_ticker

        # First call - should fetch
        info1 = self.fetcher.get_stock_info(ticker)
        self.assertEqual(info1['symbol'], 'NVDA')
        self.assertEqual(mock_yf.Ticker.call_count, 1)

        # Second call - should use cache
        info2 = self.fetcher.get_stock_info(ticker)
        self.assertEqual(info2, info1)
        self.assertEqual(mock_yf.Ticker.call_count, 1) # Count should remain 1

    @patch('stock_data.yf')
    def test_fetch_batch_single_ticker(self, mock_yf):
        # Test handling of single ticker response
        tickers = ['TSLA']
        period = '1mo'

        # Create flat DataFrame (simulating single ticker response)
        data = pd.DataFrame({'Open': [100], 'Close': [105]}, index=[0])
        mock_yf.download.return_value = data

        self.fetcher.fetch_batch_data(tickers, period)

        self.assertIn(('TSLA', period), self.fetcher.cache)
        pd.testing.assert_frame_equal(self.fetcher.cache[('TSLA', period)], data)

if __name__ == '__main__':
    unittest.main()
