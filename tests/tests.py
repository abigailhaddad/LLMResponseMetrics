# Import necessary libraries and modules
from code.functions import (
    DataLoader,
    LLMUtility,
    PerturbationGenerator,
    ModelResponseGenerator,
    LLMRatingCalculator,
)  # Custom module import
import unittest
from unittest.mock import patch, mock_open, Mock  # For mocking external dependencies
import pandas as pd
import os


class TestDataLoader(unittest.TestCase):
    """
    Unit tests for the DataLoader class.

    The DataLoader class is responsible for loading data from either a CSV file or a list of prompts.
    These tests validate that the DataLoader correctly reads and processes the input data.
    """

    @patch("pandas.read_csv")
    def test_load_data_from_csv(self, mock_read_csv):
        """
        Test the load_data method for a CSV file input.

        This test uses mocking to simulate reading data from a CSV file.
        It verifies that the method correctly reads the data and creates a DataFrame.
        """
        # Setting up the mock to return a predefined DataFrame
        mock_read_csv.return_value = pd.DataFrame({"prompt": ["test prompt"]})

        # Initialize DataLoader with a dummy file path
        loader = DataLoader("dummy.csv")

        # Load data using DataLoader
        df = loader.load_data()

        # Assertions to check if the DataFrame is loaded correctly
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["prompt"], "test prompt")

    def test_load_data_from_list(self):
        """
        Test the load_data method for a list input.

        This test verifies that the DataLoader can correctly process a list of prompts
        and create a DataFrame from it.
        """
        # Define a list of prompts
        prompts = ["prompt1", "prompt2"]

        # Initialize DataLoader with the list and is_file_path set to False
        loader = DataLoader(prompts, is_file_path=False)

        # Load data using DataLoader
        df = loader.load_data()

        # Assertions to check if the DataFrame is loaded correctly
        self.assertEqual(len(df), 2)
        self.assertIn("prompt1", df["prompt"].values)


class TestLLMUtility(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data="test_api_key")
    def test_read_api_key(self, mock_file):
        # Call the method
        result = LLMUtility.read_api_key("OPENAI")

        # Construct the expected file path
        expected_file_path = os.path.join(os.getcwd(), "..", "keys", "openai_key.txt")

        # Assert the file was opened with the correct path
        mock_file.assert_called_with(expected_file_path, "r")
        self.assertEqual(result, "test_api_key")


class TestPerturbationGenerator(unittest.TestCase):
    def test_parse_model_response(self):
        """Test parsing the model response for perturbations."""
        generator = PerturbationGenerator("model", "provider")
        response = {
            "choices": [{"message": {"content": "- Perturbation 1\n- Perturbation 2"}}]
        }
        result = generator.parse_model_response(response)
        self.assertEqual(result, ["- Perturbation 1", "- Perturbation 2"])  



class TestModelResponseGeneratorStability(unittest.TestCase):
    def setUp(self):
        self.generator = ModelResponseGenerator(
            models_dict={},
            instructions="",
            max_runs=1,
            stability_threshold=3,
            similarity_calculator=None,
            keyword_match_calculator=None,
            llm_rating_calculator=None,
            temperature=0.7,
        )

    def test_stability_with_stable_scores(self):
        """Test with scores that are stable."""
        self.generator.stability_scores = {
            "similarity_score": [0.8, 0.8, 0.8],
            "keyword_score": [0.5, 0.5, 0.5],
            "llm_rating": [0.7, 0.7, 0.7],
        }
        self.assertTrue(self.generator.is_stable())

    def test_stability_with_unstable_scores(self):
        """Test with scores that are not stable."""
        self.generator.stability_scores = {
            "similarity_score": [0.8, 0.7, 0.8],
            "keyword_score": [0.5, 0.6, 0.5],
            "llm_rating": [0.7, 0.7, 0.6],
        }
        self.assertFalse(self.generator.is_stable())

    def test_stability_with_insufficient_data(self):
        """Test with insufficient data to determine stability."""
        self.generator.stability_scores = {
            "similarity_score": [0.8, 0.8],
            "keyword_score": [0.5, 0.5],
            "llm_rating": [0.7],
        }
        self.assertFalse(self.generator.is_stable())


class TestLLMRatingCalculator(unittest.TestCase):
    @patch("code.functions.LLMUtility.call_model")
    def test_rate_response(self, mock_call_model):
        """Test rating a response based on its similarity to the target answer."""
        calculator = LLMRatingCalculator(("model_name", "provider"))
        mock_call_model.return_value = {"choices": [{"message": {"content": "7"}}]}
        row = {"target_answer": "some answer", "response": "some response"}
        rating = calculator.rate_response(row)
        self.assertEqual(rating, 0.7)


class TestFullIntegration(unittest.TestCase):
    @patch("code.functions.LLMUtility.call_model")
    @patch("pandas.read_csv")
    def test_integration(self, mock_read_csv, mock_call_model):
        # Setup mock for reading CSV
        mock_read_csv.return_value = pd.DataFrame(
            {
                "prompt": ["test prompt"],
                "target_answer": ["expected answer"],
                "keywords": ["keyword1, keyword2"],
            }
        )

        # Setup mock response for call_model
        mock_call_model.return_value = {
            "choices": [{"message": {"content": "mocked response"}}]
        }

        # Instantiate DataLoader
        loader = DataLoader("dummy.csv")
        df = loader.load_data()

        # Instantiate PerturbationGenerator with mock model and provider
        perturbation_generator = PerturbationGenerator("mock_model", "mock_provider")

        # Generate perturbations
        perturbations_dict = perturbation_generator.get_perturbations_for_all_prompts(
            df["prompt"]
        )

        # Mock components for ModelResponseGenerator
        mock_similarity_calculator = Mock()
        mock_similarity_calculator.calculate_score.return_value = 0.8
        mock_keyword_match_calculator = Mock()
        mock_keyword_match_calculator.calculate_match_percent.return_value = 0.6
        mock_llm_rating_calculator = Mock()
        mock_llm_rating_calculator.rate_response.return_value = 0.4

        # Instantiate ModelResponseGenerator with mock components
        response_generator = ModelResponseGenerator(
            models_dict={"mock_model": "mock_provider"},
            instructions="Test instructions",
            max_runs=1,
            stability_threshold=3,
            similarity_calculator=mock_similarity_calculator,
            keyword_match_calculator=mock_keyword_match_calculator,
            llm_rating_calculator=mock_llm_rating_calculator,
            temperature=0.7,
        )

        # Process prompts and check results
        results = response_generator.process_prompts_with_realtime_evaluation(
            df, perturbations_dict
        )

        # Assertions to verify integration
        self.assertIsNotNone(results)
        self.assertFalse(results.empty)
        mock_read_csv.assert_called_once()
        mock_call_model.assert_called()


# Run the tests if the script is executed directly
if __name__ == "__main__":
    unittest.main()
