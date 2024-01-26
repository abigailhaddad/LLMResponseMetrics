import os
import numpy as np
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModel
import torch
import glob
import logging
import litellm
import string
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.cluster.hierarchy import linkage, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering


litellm.set_verbose = False

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataLoader:
    """
    A class for loading data into a DataFrame.

    Args:
        input_data (str or list): The input data to be loaded. If `is_file_path` is True, it should be a file path to a CSV file. Otherwise, it should be a list of prompts.
        is_file_path (bool, optional): Indicates whether `input_data` is a file path or a list of prompts. Defaults to True.

    Methods:
        load_data(): Loads the data into a DataFrame and performs some preprocessing.

    Returns:
        pandas.DataFrame: The loaded data.

    """

    def __init__(self, input_data, is_file_path=True):
        self.input_data = input_data
        self.is_file_path = is_file_path

    def load_data(self):
        """
        Loads the data into a DataFrame and performs some preprocessing.

        Returns:
            pandas.DataFrame: The loaded data.

        """
        if self.is_file_path:
            logging.info("Reading prompts from CSV file.")
            df = pd.read_csv(self.input_data, encoding="ISO-8859-1")
        else:
            logging.info("Using direct list of prompts.")
            df = pd.DataFrame(self.input_data, columns=["prompt"])

        if "keywords" in df.columns:
            df["keywords"] = df["keywords"].apply(
                lambda x: [
                    keyword.strip().lower()
                    for keyword in x.split("\n")
                    if keyword.strip()
                ]
            )
        return df


class LLMUtility:
    @staticmethod
    def read_api_key(provider: str) -> str:
        """
        Reads the API key for a given provider from environment variables.
        Args:
            provider (str): The name of the provider, e.g., 'OPENAI' or 'ANTHROPIC'.
        Returns:
            The API key as a string.
        Raises:
            EnvironmentError: If the API key environment variable does not exist.
        """
        # Construct the environment variable name for the API key
        key_var_name = f"{provider.upper()}_KEY"

        try:
            return os.environ[key_var_name]
        except KeyError:
            raise EnvironmentError(f"Environment variable '{key_var_name}' not found.")

    @staticmethod
    def call_model(model: str, messages: list, provider: str, temperature: float):
        """
        Calls the specified model using the provider's API key and the given parameters.
        Args:
        model (str): Model identifier.
        messages (list): List of message dictionaries to send
        provider (str): The provider of the API ('OPENAI' or 'ANTHROPIC').
        temperature (float): The temperature setting for the language model.

        Returns:
        The API response.
        """
        # Set the API key for the provider
        api_key = LLMUtility.read_api_key(provider)
        try:
            response = litellm.completion(
                model=model, messages=messages, temperature=temperature, api_key=api_key
            )
            logging.info(f"API call successful. Model: {model}, Provider: {provider}")
            return response
        except Exception as e:
            logging.error(
                f"API call failed. Model: {model}, Provider: {provider}, Error: {e}"
            )
            return None


class PerturbationGenerator:
    """
    This class is responsible for generating textual perturbations for given prompts using a specified perturbation model (LLM).

    Attributes:
        perturbation_model (str): Identifier for the perturbation model to be used.
        provider (str): The provider of the perturbation model.
        temperature (float): The temperature parameter influencing the variability of generated perturbations.
        num_perturbations (float) : The number of perturbations to generate.

    Methods:
        get_perturbations(prompt): Generates and returns a list of perturbed versions of the given prompt.
        parse_model_response(response): Extracts and formats perturbations from the model's response.
        get_perturbations_for_all_prompts(prompts): Generates perturbations for a list of prompts and returns a dictionary mapping each prompt to its perturbations.
    """

    def __init__(self, model, provider, num_perturbations):
        self.perturbation_model = model
        self.provider = provider
        self.temperature = 0
        self.num_perturbations = num_perturbations

    def get_perturbations(self, prompt, rephrase_level=None):
        """
        Generates perturbations for a given prompt, with an optional rephrasing level.

        Args:
            prompt (str): The prompt for which perturbations need to be generated.
            n (int): The number of perturbations to generate.
            rephrase_level (str, optional): Level of rephrasing - None, 'moderate', or 'extensive'.

        Returns:
            list: A list of perturbations for the given prompt.
        """
        n = self.num_perturbations
        # Prepare the instruction based on the rephrase level
        rephrase_instruction = f"Generate {n} different ways to express"
        if rephrase_level:
            rephrase_instruction += f" [{rephrase_level} rephrasing]"
        rephrase_instruction += f' "{prompt}"'

        # Call the model to generate perturbations
        messages = [{"role": "user", "content": rephrase_instruction}]
        response = LLMUtility.call_model(
            self.perturbation_model, messages, self.provider, self.temperature
        )

        # Parse the model response
        perturbations = self.parse_model_response(response)

        # Check if the number of perturbations is correct
        if len(perturbations) != n:
            print(
                f"Warning: Number of perturbations for prompt '{prompt}' is {len(perturbations)}. Expected {n}"
            )
            print("Response content:")
            print(response["choices"][0]["message"]["content"])

        return perturbations

    def parse_model_response(self, response):
        """
        Parses the response from the perturbation model.

        Args:
            response (dict): The response from the perturbation model.

        Returns:
            list: A list of perturbations parsed from the response.
        """
        content = response["choices"][0]["message"]["content"]
        # Handling both numbered list and bullet points
        if content.startswith("1.") or content.strip().startswith("-"):
            # Split by newline and strip leading/trailing spaces and list markers
            perturbations = [
                pert.strip("* ").strip() for pert in content.split("\n") if pert.strip()
            ]
        else:
            # Fallback in case of unexpected formatting
            perturbations = [content.strip()]
        return perturbations

    def get_perturbations_for_all_prompts(self, prompts):
        """
        Generates perturbations for multiple prompts.

        Args:
            prompts (list): A list of prompts for which perturbations need to be generated.

        Returns:
            dict: A dictionary mapping each prompt to a list of perturbations.
        """
        perturbations_dict = {}
        for prompt in prompts:
            perturbations = [prompt]  # Include the original prompt
            paraphrased_perturbations = self.get_perturbations(prompt)
            perturbations.extend(paraphrased_perturbations)

            perturbations_dict[prompt] = perturbations
        return perturbations_dict


class ModelResponseGenerator:
    """
    Generates responses for given prompts using specified models. It also performs stability checks and real-time evaluation of responses.

    Args:
        models_dict (dict): Maps model names to their respective providers.
        instructions (str): Instructions to be included in the message content sent to the models.
        max_runs (int): Maximum number of runs to generate responses for each prompt.
        stability_threshold (int): Number of consecutive runs without score change required to consider a prompt's scores as stable.
        similarity_calculator (SimilarityCalculator): Instance of SimilarityCalculator for computing similarity scores.
        keyword_match_calculator (KeywordMatchCalculator): Instance of KeywordMatchCalculator for keyword analysis.
        llm_rating_calculator (LLMRatingCalculator): Instance of LLMRatingCalculator for rating model responses.
        temperature (float): Temperature setting for the language model, influencing response variability.

    Methods:
        is_stable(): Checks if the maximum scores have been stable over the last 'stability_threshold' runs.
        process_prompts_with_realtime_evaluation(df, perturbations_dict): Processes prompts with real-time evaluation and stability checks, and returns a DataFrame with results.
    """

    def __init__(
        self,
        models_dict,
        instructions,
        max_runs,
        stability_threshold,
        similarity_calculator,
        keyword_match_calculator,
        llm_rating_calculator,
        temperature,
    ):
        self.models_dict = models_dict
        self.instructions = instructions
        self.max_runs = max_runs
        self.similarity_calculator = similarity_calculator
        self.keyword_match_calculator = keyword_match_calculator
        self.llm_rating_calculator = llm_rating_calculator
        self.temperature = temperature
        self.stability_threshold = (
            stability_threshold  # Number of runs to check for stability
        )
        self.stability_scores = {
            "similarity_score": [],
            "keyword_score": [],
            "llm_rating": [],
        }  # To track scores

    def is_stable(self):
        """
        Checks if the maximum scores have been stable over the last 'n' runs.
        Stability is defined as the highest scores not changing for the last 'n' runs.
        """
        if any(
            len(scores) < self.stability_threshold
            for scores in self.stability_scores.values()
        ):
            return False  # Not enough data to determine stability

        for metric, scores in self.stability_scores.items():
            # Only consider the last 'n' scores for this metric
            last_n_scores = scores[-self.stability_threshold :]
            max_score = max(last_n_scores)
            if not all(score == max_score for score in last_n_scores):
                logging.info(
                    f"Scores for metric '{metric}' have not stabilized in the last {self.stability_threshold} runs. Scores: {last_n_scores}"
                )
                return False

        logging.info("Scores have stabilized across all metrics.")
        return True

    def process_prompts_with_realtime_evaluation(self, df, perturbations_dict):
        all_results = []
        for index, row in df.iterrows():
            # Reset stability scores for each new prompt
            self.stability_scores = {
                "similarity_score": [],
                "keyword_score": [],
                "llm_rating": [],
            }

            prompt = row["prompt"]
            target_answer = row.get("target_answer", None)
            keywords = row.get("keywords", None)
            perturbations = perturbations_dict.get(prompt, [prompt])

            for model, provider in self.models_dict.items():
                for run_number in range(self.max_runs):
                    actual_prompt = random.choice(perturbations)
                    temp_value = (
                        random.uniform(0.0, 1.0)
                        if self.temperature == "variable"
                        else self.temperature
                    )
                    message_content = f"{self.instructions} {actual_prompt}"
                    response = LLMUtility.call_model(
                        model,
                        [{"role": "user", "content": message_content}],
                        provider,
                        temp_value,
                    )["choices"][0]["message"]["content"]
                    # Evaluate the response
                    (
                        response_embedding,
                        similarity_score,
                    ) = self.similarity_calculator.calculate_embeddings_and_scores(
                        target_answer, response
                    )
                    keyword_score = (
                        self.keyword_match_calculator.calculate_match_percent(
                            keywords, response
                        )
                    )
                    llm_rating = self.llm_rating_calculator.rate_response(
                        {"target_answer": target_answer, "response": response}
                    )

                    self.stability_scores["similarity_score"].append(
                        max(
                            similarity_score,
                            self.stability_scores["similarity_score"][-1]
                            if self.stability_scores["similarity_score"]
                            else 0,
                        )
                    )
                    self.stability_scores["keyword_score"].append(
                        max(
                            keyword_score,
                            self.stability_scores["keyword_score"][-1]
                            if self.stability_scores["keyword_score"]
                            else 0,
                        )
                    )
                    self.stability_scores["llm_rating"].append(
                        max(
                            llm_rating,
                            self.stability_scores["llm_rating"][-1]
                            if self.stability_scores["llm_rating"]
                            else 0,
                        )
                    )

                    # Log the updated scores
                    logging.info(
                        f"Run {run_number}, Model {model}: Similarity - {similarity_score}, Keywords - {keyword_score}, LLM Rating - {llm_rating}"
                    )

                    result = {
                        "model": model,
                        "original_prompt": prompt,
                        "response": response,
                        "temperature": temp_value,
                        "actual_prompt": actual_prompt,
                        "run_number": run_number,
                        "similarity_score": similarity_score,
                        "response_embedding": response_embedding,
                        "keyword_score": keyword_score,
                        "llm_rating": llm_rating,
                        "keywords": keywords,
                    }

                    # Append results
                    all_results.append(result)

                    # Check for stability
                    if self.is_stable():
                        logging.info(
                            f"Stable scores achieved for prompt '{prompt}' after {run_number + 1} runs. Moving to next prompt."
                        )
                        break  # Breaks out of the innermost loop, moving to the next prompt

        return pd.DataFrame(all_results)


class ResultAggregator:
    """
    Class to aggregate results based on specified columns.

    Attributes:
        None

    Methods:
        aggregate(df, metric_name, group_by_columns, result_columns):
            Aggregates the results based on the specified columns.

    """

    def aggregate(self, df, metric_name, group_by_columns, result_columns):
        """
        Aggregates the results based on the specified columns.

        Args:
            df (pandas.DataFrame): The input DataFrame containing the results.
            metric_name (str): The name of the metric to sort the DataFrame by.
            group_by_columns (list): The columns to group the results by.
            result_columns (list): The columns to include in the aggregated results.

        Returns:
            pandas.DataFrame: The aggregated results DataFrame.

        """
        df_sorted = df.sort_values(by=metric_name, ascending=False)
        grouped = df_sorted.groupby(group_by_columns)
        best_results = grouped.head(1).rename(
            columns={col: f"best_{col}" for col in result_columns}
        )
        worst_results = grouped.tail(1).rename(
            columns={col: f"worst_{col}" for col in result_columns}
        )
        best_worst_merged = pd.merge(
            best_results,
            worst_results,
            on=group_by_columns,
            suffixes=("_best", "_worst"),
        )
        return best_worst_merged
    
    


class SimilarityCalculator:
    """
    Calculates similarity scores between target texts and actual texts using embeddings from a pre-trained language model.

    Args:
        model_name (str): The identifier of the pre-trained model to be used for generating text embeddings.

    Methods:
        calculate_score(target_texts, actual_texts): Computes and returns the similarity score between the target and actual texts.
        perform_similarity_analysis(df): Performs similarity analysis on a DataFrame containing 'target_answer' and 'response' columns, and returns the DataFrame with an added 'similarity_score' column.
        encode_texts(texts): Encodes a list of texts into embeddings using the pre-trained model.
        calculate_similarity_scores(df): Calculates and returns similarity scores for each row in a DataFrame.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer, self.model = self.get_model(self.model_name)
        self.model.eval()  # Set model to evaluation mode

    def get_model(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model

    def calculate_embeddings_and_scores(self, target_text, actual_text):
        """
        Calculates embeddings and similarity score between a target text and an actual text.

        Args:
            target_text (str): The target text.
            actual_text (str): The actual text to compare with the target.

        Returns:
            tuple: A tuple containing the embedding of the actual text and the similarity score.
        """
        target_embedding = self.encode_texts([target_text], self.model, self.tokenizer)
        actual_embedding = self.encode_texts([actual_text], self.model, self.tokenizer)
        similarity_score = 1 - cosine(target_embedding[0], actual_embedding[0])

        # Return both the embedding and the similarity score
        return actual_embedding[0].numpy().tolist(), similarity_score

    def calculate_score(self, target_texts, actual_texts):
        """
        Calculates the similarity score between target texts and actual texts.

        Args:
            target_texts (list): List of target texts.
            actual_texts (list): List of actual texts.

        Returns:
            float: Similarity score between 0 and 1.
        """
        model, tokenizer = self.get_model(self.model_name)
        target_embeddings = self.encode_texts(target_texts, model, tokenizer)
        actual_embeddings = self.encode_texts(actual_texts, model, tokenizer)
        return 1 - cosine(target_embeddings[0], actual_embeddings[0])

    def perform_similarity_analysis(self, df):
        """
        Performs similarity analysis on a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing target_answer and response columns.

        Returns:
            pandas.DataFrame: DataFrame with similarity_score column added.
        """
        if "target_answer" in df.columns:
            logging.info("Performing similarity analysis.")
            df["similarity_score"] = df.apply(
                lambda row: self.calculate_similarity(
                    [row["target_answer"]], [row["response"]]
                )
                if pd.notnull(row["target_answer"])
                else None,
                axis=1,
            )
        return df

    def encode_texts(self, texts: list, model, tokenizer):
        """
        Encodes the texts using the pre-trained model and tokenizer.

        Args:
            texts (list): List of texts to encode.
            model: Pre-trained model.
            tokenizer: Tokenizer.

        Returns:
            torch.Tensor: Encoded text embeddings.
        """
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings

    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculates the cosine similarity between two embeddings.
        """
        return 1 - cosine(embedding1, embedding2)

    def calculate_similarity_scores(self, df):
        """
        Calculates similarity scores for each row in the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing target_answer and response columns.

        Returns:
            pandas.Series: Series containing similarity scores.
        """
        return df.apply(
            lambda row: self.calculate_score([row["target_answer"]], [row["response"]])
            if pd.notnull(row["target_answer"])
            else None,
            axis=1,
        )


class KeywordMatchCalculator:
    def calculate_match_percent(self, target_keywords, actual_responses):
        if not target_keywords or not actual_responses:
            return 0
        response = actual_responses.lower()
        matched_keywords = sum(
            keyword.lower() in response for keyword in target_keywords
        )
        return matched_keywords / len(target_keywords)

    def calculate_keyword_scores(self, df):
        return df.apply(
            lambda row: self.calculate_match_percent(row["keywords"], [row["response"]])
            if "keywords" in row and row["keywords"]
            else None,
            axis=1,
        )


class LLMRatingCalculator:
    """
    Class for calculating ratings using the LLM evaluation model.

    Attributes:
        llm_evaluation_model (tuple): A tuple containing the LLM evaluation model.

    Methods:
        rate_response(row): Rates a response based on its similarity to the target answer.
        calculate_ratings(df): Calculates ratings for a DataFrame.

    """

    def __init__(self, llm_evaluation_model):
        self.llm_evaluation_model = llm_evaluation_model

    def rate_response(self, row):
        """
        Rates a response based on its similarity to the target answer.

        Args:
            row (pandas.Series): A row from a DataFrame containing 'target_answer' and 'response' columns.

        Returns:
            float: The rating of the response on a scale from 0 to 1, or None if the rating cannot be extracted.

        """
        model, provider = self.llm_evaluation_model
        try:
            rating_prompt = f"Rate the following response on an integer scale from 0 to 10 based on its similarity to the target answer. Only return an integer, with no comments or punctuation \n\nTarget Answer: {row['target_answer']}\nResponse: {row['response']}\nRating:"
            response = LLMUtility.call_model(
                model,
                [{"role": "user", "content": rating_prompt}],
                provider,
                temperature=0.7,
            )
            rating = response["choices"][0]["message"]["content"].strip()
            return int(rating) / 10
        except ValueError:
            logging.warning(f"Could not extract a valid rating from response: {rating}")
            return None

    def calculate_ratings(self, df):
        """
        Calculates ratings for a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing 'target_answer' and 'response' columns.

        Returns:
            pandas.Series: A Series containing the calculated ratings.

        """
        return df.apply(
            lambda row: self.rate_response(row)
            if pd.notnull(row["target_answer"]) and pd.notnull(row["response"])
            else None,
            axis=1,
        )


class ClosedEndedTextAnalysisPipeline:
    """
    Orchestrates the process of analyzing language model responses using perturbations, similarity calculations, keyword matching, and ratings.

    Args:
        input_data (str): Path to the input data file or a list of prompts.
        models_dict (dict): Dictionary mapping model identifiers to their respective providers.
        perturbation_model (tuple): Tuple containing the identifier and provider of the perturbation model.
        llm_evaluation_model (tuple): Tuple containing the identifier and provider of the LLM evaluation model.
        instructions (str): Instructions to be included in the message content sent to the models.
        similarity_model_name (str): Identifier of the model used for similarity calculations.
        max_runs (int): Maximum number of runs for generating responses for each prompt.
        temperature (float): Temperature setting for the language model.
        is_file_path (bool): Flag indicating whether the input_data is a file path (True) or a list of prompts (False).
        stability_threshold (int): Number of consecutive runs required for scores to be considered stable.
        num_perturbations: Number of perturbations to generate
    Methods:
        run_pipeline(): Executes the analysis pipeline, processes data, and returns a DataFrame with calculated metrics and results.
    """

    def __init__(
        self,
        input_data,
        models_dict,
        perturbation_model,
        llm_evaluation_model,
        instructions,
        similarity_model_name,
        max_runs,
        temperature,
        is_file_path,
        stability_threshold,
        num_perturbations=10,
    ):
        self.data_loader = DataLoader(input_data, is_file_path)
        self.temperature = temperature
        self.perturbation_generator = PerturbationGenerator(
            perturbation_model[0], perturbation_model[1], num_perturbations
        )

        # Create calculator instances
        self.similarity_calculator = SimilarityCalculator(similarity_model_name)
        self.keyword_match_calculator = KeywordMatchCalculator()
        self.llm_rating_calculator = LLMRatingCalculator(llm_evaluation_model)

        # Pass the calculator instances to ModelResponseGenerator
        self.response_generator = ModelResponseGenerator(
            models_dict,
            instructions,
            max_runs,
            stability_threshold,
            self.similarity_calculator,
            self.keyword_match_calculator,
            self.llm_rating_calculator,
            self.temperature,
        )

    def run_pipeline(self):
        """
        Runs the LLM analysis pipeline.

        Returns:
            pandas.DataFrame: Processed data with calculated metrics.

        """
        df = self.data_loader.load_data()
        all_prompts = df["prompt"].unique()
        perturbations_dict = (
            self.perturbation_generator.get_perturbations_for_all_prompts(all_prompts)
        )
        df_responses = self.response_generator.process_prompts_with_realtime_evaluation(
            df, perturbations_dict
        )
        del_file()
        return df_responses


def aggregate_best_scores(df, score_column):
    """
    Aggregates the best scores from a DataFrame based on a specified score column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the scores.
        score_column (str): The name of the column containing the scores.

    Returns:
        pandas.DataFrame: The DataFrame with the best scores for each group.

    """
    # Define the columns to include in the output
    relevant_columns = [
        "model",
        "original_prompt",
        "actual_prompt",
        "response",
        score_column,
    ]

    # Group by 'model' and 'original_prompt', and get the row with the best score in each group
    df_best_scores = df.loc[
        df.groupby(["model", "original_prompt"])[score_column].idxmax()
    ]

    # Select only relevant columns for output
    df_best_scores = df_best_scores[relevant_columns]

    return df_best_scores


def del_file():
    """
    Deletes all files with the prefix 'litellm_' in the current directory.
    """
    temp_files = glob.glob("litellm_*")
    for file in temp_files:
        os.remove(file)


class UniqueWordAnalysis:
    def __init__(self, df):
        """
        Initialize the UniqueWordAnalysis class with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the text data.
        """
        self.df = df

    def _clean_text(self, text):
        """
        Helper function to clean text by removing punctuation and converting to lowercase.

        Args:
            text (str): Text to be cleaned.

        Returns:
            set: Set of unique words in the cleaned text.
        """
        text = text.translate(str.maketrans("", "", string.punctuation))
        return set(text.lower().split())

    def add_unique_words_column(self):
        """
        Adds a column to the DataFrame with unique words in each response.

        Returns:
            pd.DataFrame: DataFrame with the new column added.
        """
        return self.df["response"].apply(self._clean_text)

    def calculate_new_unique_words_by_group(self):
        """
        Calculates new unique words for each group based on 'model' and 'original_prompt'.

        Returns:
            pd.Series: Series with new unique words for each run.
        """
        new_words_series = pd.Series(dtype=object, index=self.df.index)

        for _, group in self.df.groupby(["model", "original_prompt"]):
            seen_words = set()
            new_words_series[group.index] = group["UniqueWords"].apply(
                lambda words_set: words_set - seen_words
                or seen_words.update(words_set)
                or words_set - seen_words
            )

        return new_words_series

    def calculate_cumulative_unique_words_by_group(self):
        """
        Calculates cumulative unique words count for each group.

        Returns:
            pd.Series: Series with cumulative unique words count for each run.
        """
        cumulative_counts_series = pd.Series(dtype=int, index=self.df.index)

        for _, group in self.df.groupby(["model", "original_prompt"]):
            all_words = set()
            cumulative_counts = group["UniqueWords"].apply(
                lambda words_set: len(all_words.union(words_set))
                and all_words.update(words_set)
                or len(all_words)
            )
            cumulative_counts_series[group.index] = cumulative_counts

        return cumulative_counts_series

    def calculate_cumulative_word_percentages(self):
        """
        Calculates cumulative word percentages for each run.

        Returns:
            pd.Series: Series with cumulative word percentages for each run.
        """
        # Group by 'model' and 'original_prompt', then apply a custom function to calculate percentages
        cumulative_percentage_series = (
            self.df.groupby(["model", "original_prompt"])
            .apply(
                lambda group: group["CumulativeUniqueWords"]
                / group["CumulativeUniqueWords"].max()
                * 100
            )
            .fillna(0)
        )

        # Flatten the multi-index series to match the original DataFrame's index
        cumulative_percentage_series = cumulative_percentage_series.reset_index(
            level=[0, 1], drop=True
        )

        return cumulative_percentage_series


class ClusterAnalysis:
    def __init__(self, df):
        """
        Initialize the ClusterAnalysis class with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the data for cluster analysis.
        """
        self.df = df

    def intra_cluster_distance(self, embeddings_col, label_col):
        """
        Calculates the average intra-cluster distance.

        Args:
            embeddings_col (str): Column name for embeddings.
            label_col (str): Column name for labels.

        Returns:
            float: Average intra-cluster distance.
        """
        unique_labels = self.df[label_col].unique()
        intra_distances = []

        for label in unique_labels:
            cluster_embeddings = np.array(
                self.df[self.df[label_col] == label][embeddings_col].tolist()
            )
            distances = [
                cosine(cluster_embeddings[i], cluster_embeddings[j])
                for i in range(len(cluster_embeddings))
                for j in range(i + 1, len(cluster_embeddings))
            ]
            intra_distances.extend(distances)

        return np.mean(intra_distances)

    def inter_cluster_distance(self, embeddings_col, label_col):
        """
        Calculates the average inter-cluster distance.

        Args:
            embeddings_col (str): Column name for embeddings.
            label_col (str): Column name for labels.

        Returns:
            float: Average inter-cluster distance.
        """
        unique_labels = self.df[label_col].unique()
        inter_distances = []

        for i, label_i in enumerate(unique_labels):
            for j, label_j in enumerate(unique_labels):
                if i < j:
                    embeddings_i = np.array(
                        self.df[self.df[label_col] == label_i][embeddings_col].tolist()
                    )
                    embeddings_j = np.array(
                        self.df[self.df[label_col] == label_j][embeddings_col].tolist()
                    )
                    distances = [
                        cosine(embeddings_i[k], embeddings_j[l])
                        for k in range(len(embeddings_i))
                        for l in range(len(embeddings_j))
                    ]
                    inter_distances.extend(distances)

        return np.mean(inter_distances)

    def perform_agglomerative_clustering(self, embeddings_col, num_clusters):
        """
        Performs agglomerative clustering on the given embeddings.

        Args:
            embeddings_col (str): Column name for embeddings.
            num_clusters (int): Number of clusters.

        Returns:
            np.array: Array of cluster labels.
        """
        embeddings = np.array(self.df[embeddings_col].tolist())
        cluster_model = AgglomerativeClustering(
            n_clusters=num_clusters, metric="euclidean", linkage="ward"
        )
        labels = cluster_model.fit_predict(embeddings)
        return labels

    def plot_dendrogram(self, embeddings_col):
        """
        Plots a dendrogram for hierarchical clustering.

        Args:
            embeddings_col (str): Column name for embeddings.
        """
        embeddings = np.array(self.df[embeddings_col].tolist())
        Z = linkage(embeddings, method="ward")
        fig = ff.create_dendrogram(embeddings, linkagefun=lambda x: linkage(x, "ward"))
        fig.update_layout(
            width=800,
            height=500,
            title_text="Hierarchical Clustering Dendrogram",
            xaxis_title="Number of points in node (or index of point if no parenthesis)",
        )
        fig.show()


class Visualization:
    def __init__(self, df):
        """
        Initialize the Visualization class with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the data for visualization.
        """
        self.df = df

    def create_line_plot(
        self,
        x_column,
        y_column,
        title,
        y_title,
        color_discrete_sequence=px.colors.qualitative.Bold,
    ):
        """
        Creates a line plot with the specified parameters.

        Args:
            x_column (str): The name of the column to be used for the x-axis.
            y_column (str): The name of the column to be used for the y-axis.
            title (str): The title of the plot.
            y_title (str): The title of the y-axis.
            color_discrete_sequence (list): List of colors for distinct lines.

        Returns:
            None: This method will display the plot.
        """
        fig = px.line(
            self.df,
            x=x_column,
            y=y_column,
            color="original_prompt",
            title=title,
            color_discrete_sequence=color_discrete_sequence,
        )

        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(
                title="Run Number",
                showline=True,
                showgrid=False,
                linecolor="black",
                tickformat=",",
            ),
            yaxis=dict(
                title=y_title,
                showgrid=True,
                gridcolor="lightgray",
                linecolor="black",
                tickformat=",",
            ),
            legend=dict(
                orientation="h",
                y=-0.3,
                yanchor="top",
                x=0.5,
                xanchor="center",
                title_text="",
            ),
            margin=dict(l=20, r=20, t=60, b=40),
            title_x=0.5,
        )

        fig.update_traces(line=dict(width=2), marker=dict(size=7, opacity=0.7))
        fig.update_layout(
            title_font=dict(size=18, family="Arial, sans-serif", color="black"),
            font=dict(family="Helvetica, sans-serif", size=12, color="black"),
        )
        fig.show()
    
    def plot_distance_violin(self, closest_distances_df, distance_type_columns=["closest_intra_distance", "closest_inter_distance"]):
        """
        Creates a violin plot for displaying distribution of distances in Plotly.

        Args:
            closest_distances_df (pd.DataFrame): DataFrame containing distance data.
            distance_type_columns (list): List of columns for distance types to plot.

        Returns:
            None: This method will display the plot.
        """
        melted_df = closest_distances_df.melt(id_vars=["response"], value_vars=distance_type_columns, var_name="Distance Type", value_name="Distance")

        melted_df["Distance Type"] = melted_df["Distance Type"].replace({
        "closest_intra_distance": "Intra-label",
        "closest_inter_distance": "Inter-label",
        })

        # Create the violin plot
        fig = px.violin(melted_df, x="Distance Type", y="Distance", box=False, points=False, color="Distance Type")

        # Add swarmplot-like markers
        for dist_type in melted_df['Distance Type'].unique():
            dist_df = melted_df[melted_df['Distance Type'] == dist_type]
            fig.add_trace(go.Scatter(
            x=dist_df['Distance Type'], 
            y=dist_df['Distance'],
            mode='markers',
            marker=dict(color='black', size=4),
            name=f'{dist_type} points'
            ))

        # Update layout
        fig.update_layout(
            title="Density and Distribution of Closest Intra-Label and Inter-Label Distances",
            xaxis_title="Distance Type",
            yaxis_title="Distance",
            width=800,
            height=600
            )

        fig.show()

    def plot_optimal_cluster_heatmap(
        self, cluster_label_column="cluster_level_optimal", label_column="label"
    ):
        """
        Creates a heatmap for visualizing the distribution of clusters using Plotly.

        Args:
            cluster_label_column (str): Column name for cluster labels.
            label_column (str): Column name for original labels.

        Returns:
            None: This method will display the plot.
        """
        ct = pd.crosstab(self.df[label_column], self.df[cluster_label_column])

        # Create the heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=ct.values,
                x=ct.columns,
                y=ct.index,
                colorscale="YlGnBu",
                colorbar=dict(title="Count"),
            )
        )

        # Update layout
        fig.update_layout(
            title="Cross-tabulation of Original Labels and Optimal Hierarchical Clusters",
            xaxis_title="Optimal Cluster",
            yaxis_title="Original Label",
            width=800,
            height=600,
        )

        fig.show()


class EmbeddingAnalysis:
    def __init__(self, df, embeddings_col):
        """
        Initialize the EmbeddingAnalysis class with a DataFrame and the name of the embeddings column.

        Args:
            df (pd.DataFrame): DataFrame containing the data for analysis.
            embeddings_col (str): Column name containing embeddings data.
        """
        self.df = df
        self.embeddings_col = embeddings_col

    def find_closest_distances_per_response(self, label_col):
        """
        Finds the closest intra- and inter-cluster distances for each response.

        Args:
            label_col (str): Column name for the labels.

        Returns:
            pd.DataFrame: DataFrame with closest distances per response.
        """
        embeddings = np.array(self.df[self.embeddings_col].tolist())
        labels = self.df[label_col].to_numpy()
        pairwise_distances = squareform(pdist(embeddings, metric="cosine"))
        closest_distances = []

        for i, label in enumerate(labels):
            # Calculate intra- and inter-cluster distances
            intra_distances = pairwise_distances[i][labels == label]
            intra_distances[intra_distances == 0] = np.inf  # Exclude distance to itself
            inter_distances = pairwise_distances[i][labels != label]

            # Find closest distances
            closest_intra = (
                np.min(intra_distances) if intra_distances.size > 0 else np.inf
            )
            closest_inter = (
                np.min(inter_distances) if inter_distances.size > 0 else np.inf
            )

            closest_distances.append(
                {
                    "response": self.df.iloc[i]["response"],
                    "closest_intra_distance": closest_intra,
                    "closest_inter_distance": closest_inter,
                }
            )

        return pd.DataFrame(closest_distances)

    def optimal_number_of_clusters(self):
        """
        Determines the optimal number of clusters using hierarchical clustering.

        Returns:
            int: Optimal number of clusters.
        """
        embeddings = np.array(self.df[self.embeddings_col].tolist())
        # [Code for calculating the optimal number of clusters]

        Z = linkage(embeddings, method="ward")

        # Retrieve the last ten distances
        last = Z[-10:, 2]
        last_rev = last[::-1]

        # Compute the acceleration (second derivative)
        acceleration = np.diff(last, 2)
        acceleration_rev = acceleration[::-1]

        # Find the number of clusters where the acceleration is the highest
        k = acceleration_rev.argmax() + 2

        return k

    def create_cluster_levels_automatically(self):
        """
        Creates cluster levels automatically based on hierarchical clustering.

        Returns:
            pd.DataFrame: DataFrame with additional cluster level columns.
        """
        embeddings = np.array(self.df[self.embeddings_col].tolist())
        # [Code for creating cluster levels]

        Z = linkage(embeddings, method="ward")

        # Find significant levels by looking at large increases in merge distances
        distances = Z[:, 2]
        distance_diffs = np.diff(distances)
        threshold = np.percentile(
            distance_diffs, 75
        )  # Using the 75th percentile as a threshold

        # Identify significant increases
        significant_increases = distance_diffs > threshold
        levels = distances[:-1][significant_increases]

        # Reverse the levels so that largest clusters are labeled as level_1
        reversed_levels = levels[::-1]

        for idx, level in enumerate(reversed_levels, start=1):
            self.df[f"cluster_level_{idx}"] = fcluster(Z, level, criterion="distance")

        return self.df

    def get_cluster_samples(self, samples_per_cluster=1):
        """
        Gets random samples from each cluster.

        Args:
            samples_per_cluster (int): Number of samples per cluster.

        Returns:
            dict: Dictionary with cluster samples for each cluster level.
        """
        cluster_samples = {}

        cluster_cols = [
            col for col in self.df.columns if col.startswith("cluster_level_")
        ]
        cluster_samples = {}

        for cluster_col in cluster_cols:
            cluster_samples[cluster_col] = self.hierarchical_random_sample(
                cluster_col, samples_per_cluster
            )

        return cluster_samples

    def hierarchical_random_sample(self, cluster_column, samples_per_cluster=1):
        """
        Samples a specified number of entries from each cluster.

        Args:
            cluster_column (str): Column name representing cluster labels.
            samples_per_cluster (int): Number of samples per cluster.

        Returns:
            pd.DataFrame: DataFrame containing the sampled entries.
        """
        return (
            self.df.groupby(cluster_column)
            .apply(lambda x: x.sample(samples_per_cluster))
            .reset_index(drop=True)
        )


class PairwiseAnalysis:
    def __init__(self, df, embeddings_col, label_col):
        """
        Initialize the PairwiseAnalysis class with a DataFrame, embeddings column, and label column.

        Args:
            df (pd.DataFrame): DataFrame containing the data for analysis.
            embeddings_col (str): Column name containing embeddings data.
            label_col (str): Column name containing label data.
        """
        self.df = df
        self.embeddings_col = embeddings_col
        self.label_col = label_col

    def find_closest_pairs_with_labels(self, n=5, debug=False):
        """
        Finds the closest pairs of responses within and across clusters.

        Args:
            n (int): Number of closest pairs to find.
            debug (bool): If True, enables debug mode for additional output.

        Returns:
            pd.DataFrame: DataFrame with closest pairs information.
        """
        embeddings = np.array(self.df[self.embeddings_col].tolist())
        labels = self.df[self.label_col].to_numpy()

        # Calculate pairwise distances
        pairwise_distances = squareform(pdist(embeddings, metric="cosine"))

        closest_pairs_info = []

        for i, label in enumerate(labels):
            # Intra-cluster distances
            intra_cluster_indices = np.where(labels == label)[0]
            intra_distances = pairwise_distances[i, intra_cluster_indices]
            intra_distances[
                intra_cluster_indices == i
            ] = np.inf  # Exclude distance to itself

            # Inter-cluster distances
            inter_cluster_indices = np.where(labels != label)[0]
            inter_distances = pairwise_distances[i, inter_cluster_indices]

            # Find the closest intra-cluster response
            if len(intra_distances) > 0 and not all(intra_distances == np.inf):
                closest_intra_idx = intra_cluster_indices[np.argmin(intra_distances)]
                closest_intra_distance = np.min(intra_distances)

                closest_pairs_info.append(
                    {
                        "response": self.df.iloc[i]["response"],
                        "response_label": label,
                        "closest_response": self.df.iloc[closest_intra_idx]["response"],
                        "closest_label": self.df.iloc[closest_intra_idx][
                            self.label_col
                        ],
                        "distance": closest_intra_distance,
                        "type": "Intra-cluster",
                    }
                )

            # Find the closest inter-cluster response
            if len(inter_distances) > 0:
                closest_inter_idx = inter_cluster_indices[np.argmin(inter_distances)]
                closest_inter_distance = np.min(inter_distances)

                closest_pairs_info.append(
                    {
                        "response": self.df.iloc[i]["response"],
                        "response_label": label,
                        "closest_response": self.df.iloc[closest_inter_idx]["response"],
                        "closest_label": self.df.iloc[closest_inter_idx][
                            self.label_col
                        ],
                        "distance": closest_inter_distance,
                        "type": "Inter-cluster",
                    }
                )

        return pd.DataFrame(closest_pairs_info)


class OpenEndedTextAnalysisPipeline:
    def __init__(
        self,
        prompts,
        models_dict,
        instructions,
        temperature,
        stability_criteria,
        stability_threshold,
        similarity_calculator,
        perturbation_model,
        provider,
        num_perturbations=10,
    ):
        """
        Initialize the OpenEndedTextAnalysisPipeline class.

        Args:
            prompts (list): List of prompts to process.
            models_dict (dict): Dictionary mapping model names to their providers.
            instructions (str): Instructions for generating responses.
            temperature (float): Temperature setting for the language model.
            stability_criteria (float): Stability criteria for the generated responses.
            stability_threshold (int): Number of runs to consider for stability checking.
            similarity_calculator (SimilarityCalculator): Instance of SimilarityCalculator.
            perturbation_model (str): Model name for generating perturbations.
            provider (str): Provider of the perturbation model.
            num_perturbations (int): Number of perturbations to generate per prompt.
        """
        self.prompts = prompts
        self.models_dict = models_dict
        self.instructions = instructions
        self.temperature = temperature
        self.stability_criteria = stability_criteria
        self.stability_threshold = stability_threshold
        self.similarity_calculator = similarity_calculator
        self.perturbation_model = perturbation_model
        self.provider = provider
        self.num_perturbations = num_perturbations

    def process_all_prompts_models(self):
        """
        Processes all prompts and generates responses.

        Returns:
            pd.DataFrame: DataFrame containing the generated responses.
        """
        perturbation_generator = PerturbationGenerator(
            self.perturbation_model, self.provider, self.num_perturbations
        )
        perturbations_dict = perturbation_generator.get_perturbations_for_all_prompts(
            self.prompts
        )

        all_responses = []
        for model, model_provider in self.models_dict.items():
            for prompt in self.prompts:
                responses = self.generate_responses_for_prompt(
                    model, model_provider, prompt, perturbations_dict
                )
                all_responses.extend(responses)
        return pd.DataFrame(all_responses)

    def generate_responses_for_prompt(
        self, model, provider, prompt, perturbations_dict
    ):
        """
        Generates responses for a single prompt.

        Args:
            model (str): Model name.
            provider (str): Provider name.
            prompt (str): The prompt to generate a response for.
            perturbations_dict (dict): Dictionary of perturbations.

        Returns:
            list: List of response dictionaries.
        """
        responses = []
        previous_embeddings = []
        stable_runs = 0
        perturbed_prompts = perturbations_dict.get(prompt, [prompt])

        for run_number in range(self.stability_threshold):
            actual_prompt = random.choice(perturbed_prompts)
            temp_value = (
                random.uniform(0.0, 1.0)
                if self.temperature == "variable"
                else self.temperature
            )
            response_content = LLMUtility.call_model(
                model,
                [{"role": "user", "content": f"{self.instructions} {actual_prompt}"}],
                provider,
                temp_value,
            )["choices"][0]["message"]["content"]

            current_embedding = Utilities.get_response_embedding(
                response_content, self.similarity_calculator
            )

            if previous_embeddings:
                similarities = [
                    self.similarity_calculator.calculate_similarity(
                        embed, current_embedding
                    )
                    for embed in previous_embeddings
                ]
                max_similarity = max(similarities) if similarities else 0
                if max_similarity >= self.stability_criteria:
                    stable_runs += 1
                else:
                    stable_runs = 0

                if stable_runs >= self.stability_threshold:
                    break
            else:
                max_similarity = 0

            previous_embeddings.append(current_embedding)

            responses.append(
                {
                    "model": model,
                    "original_prompt": prompt,
                    "actual_prompt": actual_prompt,
                    "response": response_content,
                    "temperature": temp_value,
                    "run_number": run_number + 1,
                    "similarity": max_similarity,
                    "ResponseEmbedding": current_embedding,
                }
            )

        return responses

    def apply_additional_metrics(self, df_responses):
        """
        Applies additional metrics to the DataFrame of responses.

        Args:
            df_responses (pd.DataFrame): DataFrame containing the responses.

        Returns:
            pd.DataFrame: DataFrame with additional metrics applied.
        """
        # Apply unique words and cumulative metrics
        unique_word_analysis = UniqueWordAnalysis(df_responses)
        df_responses["UniqueWords"] = unique_word_analysis.add_unique_words_column()
        df_responses[
            "CumulativeUniqueWords"
        ] = unique_word_analysis.calculate_cumulative_unique_words_by_group()
        df_responses[
            "NewUniqueWords"
        ] = unique_word_analysis.calculate_new_unique_words_by_group()
        df_responses[
            "CumulativeWordPercentage"
        ] = unique_word_analysis.calculate_cumulative_word_percentages()

        return df_responses

    def run_pipeline(self):
        """
        Executes the entire text analysis pipeline.

        Returns:
            pd.DataFrame: DataFrame containing the analyzed and processed data.
        """
        df_responses = self.process_all_prompts_models()
        df_final = self.apply_additional_metrics(df_responses)
        return df_final


class Utilities:
    @staticmethod
    def get_response_embedding(response, similarity_calculator):
        """
        Generates an embedding for a given response.

        Args:
            response (str): The response to encode.
            similarity_calculator (SimilarityCalculator): The similarity calculator with model and tokenizer.

        Returns:
            list: List representation of the response embedding.
        """
        model = similarity_calculator.model
        tokenizer = similarity_calculator.tokenizer
        embedding = similarity_calculator.encode_texts([response], model, tokenizer)
        return embedding[0].numpy().tolist()
