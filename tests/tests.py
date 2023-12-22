from code.functions import DataLoader
import unittest
from unittest.mock import patch
import pandas as pd

class TestDataLoader(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load_data_from_csv(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({'prompt': ['test prompt']})
        loader = DataLoader('dummy.csv')
        df = loader.load_data()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['prompt'], 'test prompt')

    def test_load_data_from_list(self):
        prompts = ['prompt1', 'prompt2']
        loader = DataLoader(prompts, is_file_path=False)
        df = loader.load_data()
        self.assertEqual(len(df), 2)
        self.assertIn('prompt1', df['prompt'].values)

# Run the tests
if __name__ == '__main__':
    unittest.main()
