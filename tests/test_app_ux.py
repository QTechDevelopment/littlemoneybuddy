import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()
import app

class TestUXChanges(unittest.TestCase):
    def test_parse_excel_tickers_warning(self):
        # Create a mock dataframe with 51 tickers
        tickers = [f"TICK{i}" for i in range(51)]
        df = pd.DataFrame({'ticker': tickers})

        # Mock uploaded file
        mock_file = MagicMock()

        # Patch pandas read_excel to return our dataframe
        with patch('pandas.read_excel', return_value=df):
            result = app.parse_excel_tickers(mock_file)

            # Verify result length is capped at 50
            self.assertEqual(len(result), 50)

            # Verify warning was called
            # We access the mock streamlit module we injected
            app.st.warning.assert_called_once()
            args, _ = app.st.warning.call_args
            self.assertIn("analyzing first 50 stocks", args[0])
            self.assertIn("out of 51 found", args[0])

    def test_parse_excel_tickers_no_warning_under_limit(self):
        # Create a mock dataframe with 49 tickers
        tickers = [f"TICK{i}" for i in range(49)]
        df = pd.DataFrame({'ticker': tickers})

        # Mock uploaded file
        mock_file = MagicMock()

        # Patch pandas read_excel to return our dataframe
        with patch('pandas.read_excel', return_value=df):
            app.st.warning.reset_mock()
            result = app.parse_excel_tickers(mock_file)

            # Verify result length
            self.assertEqual(len(result), 49)

            # Verify warning was NOT called
            app.st.warning.assert_not_called()

    def test_parse_excel_tickers_exact_limit(self):
        # Create a mock dataframe with exactly 50 tickers (boundary condition)
        tickers = [f"TICK{i}" for i in range(50)]
        df = pd.DataFrame({'ticker': tickers})

        # Mock uploaded file
        mock_file = MagicMock()

        # Patch pandas read_excel to return our dataframe
        with patch('pandas.read_excel', return_value=df):
            app.st.warning.reset_mock()
            result = app.parse_excel_tickers(mock_file)

            # Verify result length
            self.assertEqual(len(result), 50)

            # Verify warning was NOT called
            app.st.warning.assert_not_called()

if __name__ == '__main__':
    unittest.main()
