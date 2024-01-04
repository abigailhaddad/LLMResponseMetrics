import os
import pandas as pd
import random
from litellm import completion
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import glob
import logging
import litellm

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
        Reads the API key for a given provider from a file located in a 'keys' directory
        above the current working directory.

        Args:
        provider (str): The name of the provider, e.g., 'OPENAI' or 'ANTHROPIC'.

        Returns:
        The API key as a string.

        Raises:
        FileNotFoundError: If the API key file does not exist.
        """
        # Construct the file path for the API key file
        key_filename = f"{provider.lower()}_key.txt"
        # Adjust the path to look for the 'keys' folder at the same level as the script directory
        key_folder_path = os.path.join(os.getcwd(), "..", "keys")
        key_file_path = os.path.join(key_folder_path, key_filename)

        try:
            with open(key_file_path, "r") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"API key file '{key_filename}' not found in '{key_folder_path}'."
            )

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
            response = completion(
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

    Methods:
        get_perturbations(prompt): Generates and returns a list of perturbed versions of the given prompt.
        parse_model_response(response): Extracts and formats perturbations from the model's response.
        get_perturbations_for_all_prompts(prompts): Generates perturbations for a list of prompts and returns a dictionary mapping each prompt to its perturbations.
    """

    def __init__(self, model, provider):
        self.perturbation_model = model
        self.provider = provider
        self.temperature = 0

    def get_perturbations(self, prompt, n=10, rephrase_level=None):
        """
        Generates perturbations for a given prompt, with an optional rephrasing level.

        Args:
            prompt (str): The prompt for which perturbations need to be generated.
            n (int): The number of perturbations to generate.
            rephrase_level (str, optional): Level of rephrasing - None, 'moderate', or 'extensive'.

        Returns:
            list: A list of perturbations for the given prompt.
        """
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


class LLMAnalysisPipeline:
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
    ):
        self.data_loader = DataLoader(input_data, is_file_path)
        self.temperature = temperature
        self.perturbation_generator = PerturbationGenerator(
            perturbation_model[0], perturbation_model[1]
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
