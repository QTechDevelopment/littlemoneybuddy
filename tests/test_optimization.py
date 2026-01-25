
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
        # Tickers are passed as first positional argument (list)
        self.assertIn('AAPL', args[0])

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

    @patch('stock_data.yf')
    def test_fetch_batch_handles_exception(self, mock_yf):
        # Test that exceptions during batch fetch are handled gracefully
        tickers = ['AAPL', 'MSFT']
        period = '1mo'
        
        # Make download raise an exception
        mock_yf.download.side_effect = Exception("API error")
        
        # Should not raise, just print error
        self.fetcher.fetch_batch_data(tickers, period)
        
        # Cache should remain empty
        self.assertNotIn(('AAPL', period), self.fetcher.cache)
        self.assertNotIn(('MSFT', period), self.fetcher.cache)

    @patch('stock_data.yf')
    def test_fetch_batch_handles_empty_dataframe(self, mock_yf):
        # Test handling of empty DataFrame response
        tickers = ['INVALID1', 'INVALID2']
        period = '1mo'
        
        # Return empty DataFrame
        mock_yf.download.return_value = pd.DataFrame()
        
        self.fetcher.fetch_batch_data(tickers, period)
        
        # Cache should not be populated
        self.assertNotIn(('INVALID1', period), self.fetcher.cache)
        self.assertNotIn(('INVALID2', period), self.fetcher.cache)

    @patch('stock_data.yf')
    def test_fetch_batch_partial_success(self, mock_yf):
        # Test scenario where some tickers return data and others don't
        tickers = ['AAPL', 'INVALID', 'MSFT']
        period = '1mo'
        
        # Create DataFrame with only AAPL and MSFT data
        columns = pd.MultiIndex.from_product([['AAPL', 'MSFT'], ['Open', 'Close', 'High', 'Low', 'Volume']], names=['Ticker', 'Price'])
        data = pd.DataFrame(index=range(5), columns=columns)
        data.loc[:, ('AAPL', 'Close')] = 150.0
        data.loc[:, ('MSFT', 'Close')] = 300.0
        
        mock_yf.download.return_value = data
        
        self.fetcher.fetch_batch_data(tickers, period)
        
        # Valid tickers should be cached
        self.assertIn(('AAPL', period), self.fetcher.cache)
        self.assertIn(('MSFT', period), self.fetcher.cache)
        # Invalid ticker should not be cached
        self.assertNotIn(('INVALID', period), self.fetcher.cache)

    @patch('stock_data.yf')
    def test_fetch_batch_empty_ticker_list(self, mock_yf):
        # Test with empty ticker list
        self.fetcher.fetch_batch_data([])
        
        # Should not call download
        mock_yf.download.assert_not_called()

    @patch('stock_data.yf')
    def test_fetch_batch_all_nan_data(self, mock_yf):
        # Test handling of DataFrame with all NaN values
        tickers = ['AAPL', 'MSFT']
        period = '1mo'
        
        # Create DataFrame with all NaN values
        columns = pd.MultiIndex.from_product([tickers, ['Open', 'Close', 'High', 'Low', 'Volume']], names=['Ticker', 'Price'])
        data = pd.DataFrame(index=range(5), columns=columns)
        # Leave all values as NaN
        
        mock_yf.download.return_value = data
        
        self.fetcher.fetch_batch_data(tickers, period)
        
        # Tickers with all NaN should not be cached (due to dropna and empty check)
        self.assertNotIn(('AAPL', period), self.fetcher.cache)
        self.assertNotIn(('MSFT', period), self.fetcher.cache)

if __name__ == '__main__':
    unittest.main()
