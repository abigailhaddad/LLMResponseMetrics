# Import necessary libraries and modules
from code.functions import DataLoader  # Custom module import
import unittest
from unittest.mock import patch  # For mocking external dependencies
import pandas as pd

class TestDataLoader(unittest.TestCase):
    """
    Unit tests for the DataLoader class.

    The DataLoader class is responsible for loading data from either a CSV file or a list of prompts.
    These tests validate that the DataLoader correctly reads and processes the input data.
    """

    @patch('pandas.read_csv')
    def test_load_data_from_csv(self, mock_read_csv):
        """
        Test the load_data method for a CSV file input.

        This test uses mocking to simulate reading data from a CSV file.
        It verifies that the method correctly reads the data and creates a DataFrame.
        """
        # Setting up the mock to return a predefined DataFrame
        mock_read_csv.return_value = pd.DataFrame({'prompt': ['test prompt']})
        
        # Initialize DataLoader with a dummy file path
        loader = DataLoader('dummy.csv')
        
        # Load data using DataLoader
        df = loader.load_data()
        
        # Assertions to check if the DataFrame is loaded correctly
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['prompt'], 'test prompt')

    def test_load_data_from_list(self):
        """
        Test the load_data method for a list input.

        This test verifies that the DataLoader can correctly process a list of prompts
        and create a DataFrame from it.
        """
        # Define a list of prompts
        prompts = ['prompt1', 'prompt2']
        
        # Initialize DataLoader with the list and is_file_path set to False
        loader = DataLoader(prompts, is_file_path=False)
        
        # Load data using DataLoader
        df = loader.load_data()
        
        # Assertions to check if the DataFrame is loaded correctly
        self.assertEqual(len(df), 2)
        self.assertIn('prompt1', df['prompt'].values)

# Run the tests if the script is executed directly
if __name__ == '__main__':
    unittest.main()
