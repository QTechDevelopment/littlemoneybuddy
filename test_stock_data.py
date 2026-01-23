
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from stock_data import StockDataFetcher

class TestStockDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = StockDataFetcher()
        self.ticker = "AAPL"
        self.period = "1mo"

        # Create a mock DataFrame
        self.mock_df = pd.DataFrame({
            'Close': [150.0, 151.0, 152.0],
            'Volume': [1000, 2000, 3000]
        })

        self.mock_info = {
            'symbol': 'AAPL',
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }

    @patch('yfinance.Ticker')
    def test_get_stock_data_caching(self, mock_ticker_class):
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_class.return_value = mock_ticker_instance
        mock_ticker_instance.history.return_value = self.mock_df

        # First call - should hit the API
        df1 = self.fetcher.get_stock_data(self.ticker, self.period)

        # Second call - should hit the cache
        df2 = self.fetcher.get_stock_data(self.ticker, self.period)

        # Verify calls
        mock_ticker_class.assert_called_once_with(self.ticker)
        mock_ticker_instance.history.assert_called_once_with(period=self.period)

        # Verify results are the same
        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(df1, self.mock_df)

    @patch('yfinance.Ticker')
    def test_get_stock_info_caching(self, mock_ticker_class):
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_class.return_value = mock_ticker_instance
        mock_ticker_instance.info = self.mock_info

        # First call - should hit the API
        info1 = self.fetcher.get_stock_info(self.ticker)

        # Second call - should hit the cache
        info2 = self.fetcher.get_stock_info(self.ticker)

        # Verify calls
        mock_ticker_class.assert_called_once_with(self.ticker)

        # Verify results
        self.assertEqual(info1['symbol'], self.ticker)
        self.assertEqual(info1, info2)

if __name__ == '__main__':
    unittest.main()
