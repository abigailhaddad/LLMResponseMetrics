import os
import pandas as pd
import random
import re
from litellm import completion
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import glob
import logging

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, input_data, is_file_path=True):
        self.input_data = input_data
        self.is_file_path = is_file_path

    def load_data(self):
        if self.is_file_path:
            logging.info("Reading prompts from CSV file.")
            df = pd.read_csv(self.input_data, encoding='ISO-8859-1')
        else:
            logging.info("Using direct list of prompts.")
            df = pd.DataFrame(self.input_data, columns=['prompt'])

        if 'keywords' in df.columns:
            df['keywords'] = df['keywords'].apply(lambda x: [keyword.strip().lower() for keyword in x.split('\n') if keyword.strip()])
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
        key_folder_path = os.path.join(os.getcwd(), '..', 'keys')
        key_file_path = os.path.join(key_folder_path, key_filename)
    
        try:
            with open(key_file_path, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file '{key_filename}' not found in '{key_folder_path}'.")

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
        # Call the completion endpoint with the provided parameters
        response = completion(model=model, messages=messages, temperature=temperature, api_key = api_key)
        return response


class PerturbationGenerator:
    def __init__(self, perturbation_model, num_perturbations, temperature):
        self.perturbation_model = perturbation_model[0]
        self.provider = perturbation_model[1]
        self.num_perturbations = num_perturbations
        self.temperature = temperature

    def get_perturbations(self, prompt):
        if self.num_perturbations == 0:
            return [prompt]

        paraphrase_instruction = f"Generate a bulleted list of {self.num_perturbations + 1} sentences with the same meaning as \"{prompt}\""
        messages = [{"role": "user", "content": paraphrase_instruction}]
        response = LLMUtility.call_model(self.perturbation_model, messages, self.provider, self.temperature)
        return self.parse_model_response(response)

    def parse_model_response(self, response):
        content = response['choices'][0]['message']['content']
        if content.startswith('1.'):
            perturbations = re.split(r'\n\d+\.\s*', content)
            perturbations = [pert.strip() for pert in perturbations if pert.strip()]
        else:
            perturbations = content.split('\n- ')[1:]
            perturbations = [pert.strip('* ') for pert in perturbations]
        return perturbations

    def get_perturbations_for_all_prompts(self, prompts):
        perturbations_dict = {}
        for prompt in prompts:
            perturbations_dict[prompt] = self.get_perturbations(prompt)
        return perturbations_dict


class ModelResponseGenerator:
    def __init__(self, models_dict, instructions, temperature):
        self.models_dict = models_dict
        self.instructions = instructions
        self.temperature = temperature
    
    def create_result_dict(self, model, original_prompt, actual_prompt, response, temperature, target_answer, keywords):
        generated_text = response['choices'][0]['message']['content']
        result = {
            'model': model,
            'original_prompt': original_prompt,
            'response': generated_text,
            'temperature': temperature,
            'actual_prompt': actual_prompt,
        }
        if target_answer is not None:
            result['target_answer'] = target_answer
        if keywords is not None:
            result['keywords'] = keywords
        return result

    def process_single_prompt(self, model, provider, original_prompt, actual_prompt, instructions, temperature, target_answer=None, keywords=None):
        # If temperature is "variable", randomize it; otherwise, use the fixed value
        temp_value = random.uniform(0.0, 1.0) if temperature == "variable" else temperature

        # Form the content to send to the model
        message_content = f"{instructions} {actual_prompt}"

        # Call the model and get the response
        response = LLMUtility.call_model(model, [{"role": "user", "content": message_content}], provider, temp_value)

        # Create and return the result dictionary
        return self.create_result_dict(model, original_prompt, actual_prompt, response, temp_value, target_answer, keywords)


    def process_prompts(self, df, perturbations_dict):
        results = []
        for index, row in df.iterrows():
            prompt = row['prompt']
            target_answer = row.get('target_answer', None)
            keywords = row.get('keywords', None)
            perturbations = perturbations_dict.get(prompt, [prompt])  # Use the original prompt as a fallback

            for model, provider in self.models_dict.items():
                for perturbation in perturbations:
                    results.append(self.process_single_prompt(model, provider, prompt, perturbation, self.instructions, self.temperature, target_answer, keywords))
        return pd.DataFrame(results)



class ResultAggregator:
    def aggregate(self, df, metric_name, group_by_columns, result_columns):
        df_sorted = df.sort_values(by=metric_name, ascending=False)
        grouped = df_sorted.groupby(group_by_columns)
        best_results = grouped.head(1).rename(columns={col: f'best_{col}' for col in result_columns})
        worst_results = grouped.tail(1).rename(columns={col: f'worst_{col}' for col in result_columns})
        best_worst_merged = pd.merge(best_results, worst_results, on=group_by_columns, suffixes=('_best', '_worst'))
        return best_worst_merged




class SimilarityCalculator:
    def __init__(self, model_name):
        self.model_name = model_name

    def calculate_similarity(self, target_texts: list, actual_texts: list):
        model, tokenizer = self.get_model(self.model_name)
        target_embeddings = self.encode_texts(target_texts, model, tokenizer)
        actual_embeddings = self.encode_texts(actual_texts, model, tokenizer)

        similarities = [1 - cosine(target_embedding, actual_embedding)
                        for target_embedding, actual_embedding in zip(target_embeddings, actual_embeddings)]
        return similarities

    def perform_similarity_analysis(self, df):
        if 'target_answer' in df.columns:
            logging.info("Performing similarity analysis.")
            df['similarity_score'] = df.apply(
                lambda row: self.calculate_similarity([row['target_answer']], [row['response']])
                if pd.notnull(row['target_answer']) else None, 
                axis=1
            )
        return df

    def calculate_and_aggregate(self, df):
        df_with_similarity = self.perform_similarity_analysis(df)
        result_aggregator = ResultAggregator()
        aggregated_results = result_aggregator.aggregate(
            df_with_similarity, 
            'similarity_score', 
            ['model', 'original_prompt'], 
            ['response', 'similarity_score']
        )
        return aggregated_results

    def get_model(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return model, tokenizer

    def encode_texts(self, texts: list, model, tokenizer):
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings


class KeywordMatchCalculator:
    def calculate_keyword_match_percent(self, target_keywords: list, actual_responses: list):
        """
        Calculates the fraction of target keywords found in each actual text response.

        Args:
        target_keywords (list): The list of target keywords.
        actual_responses (list): The list of actual text responses.

        Returns:
        match_fractions (list): The list of keyword match fractions.
        """
        match_fractions = []
        for response in actual_responses:
            if not target_keywords or not response:
                match_fractions.append(0)
                continue

            response_lower = response.lower()
            matched_keywords = sum(keyword.lower() in response_lower for keyword in target_keywords)
            match_fraction = matched_keywords / len(target_keywords)  # Fraction instead of percentage
            match_fractions.append(match_fraction)
        return match_fractions
    
    def add_keyword_match_percentages(self, df, target_keywords_column='keywords'):
        df['keyword_match_percent'] = df.apply(
            lambda row: self.calculate_keyword_match_percent(row[target_keywords_column], [row['response']])[0]
            if row[target_keywords_column] else None, 
            axis=1
        )
        return df

    def calculate_and_aggregate(self, df):
        df_with_keyword_matches = self.add_keyword_match_percentages(df, 'keywords')
        result_aggregator = ResultAggregator()
        aggregated_results = result_aggregator.aggregate(
            df_with_keyword_matches, 
            'keyword_match_percent', 
            ['model', 'original_prompt'], 
            ['response', 'keyword_match_percent']
        )
        return aggregated_results

class LLMRatingCalculator:
    def __init__(self, llm_evaluation_model):
        self.llm_evaluation_model = llm_evaluation_model

    def rate_response_with_llm(self, row):
        model, provider = self.llm_evaluation_model
        target_answer = row['target_answer']
        actual_response = row['response']

        rating_prompt = f"Rate the following response on an integer scale from 1 to 10 based on its similarity to the target answer. Only return an integer, with no comments or punctuation \n\nTarget Answer: {target_answer}\nResponse: {actual_response}\nRating:"

        response = LLMUtility.call_model(model, [{"role": "user", "content": rating_prompt}], provider, temperature=0.7)
        rating = response['choices'][0]['message']['content'].strip()

        try:
            return int(rating)
        except ValueError:
            logging.warning(f"Could not extract a valid rating from response: {rating}")
            return None

    def add_ratings_to_df(self, df):
        df['rating'] = df.apply(lambda row: self.rate_response_with_llm(row), axis=1)/10
        return df
    
    def calculate_and_aggregate(self, df):
        df_with_ratings = self.add_ratings_to_df(df)
        result_aggregator = ResultAggregator()
        aggregated_results = result_aggregator.aggregate(
            df_with_ratings, 
            'rating', 
            ['model', 'original_prompt'], 
            ['response', 'rating']
        )
        return aggregated_results


class LLMAnalysisPipeline:
    def __init__(self, input_data, models_dict, perturbation_model, llm_evaluation_model, temperature, 
    num_runs, is_file_path, similarity_model_name, num_perturbations, instructions):
        self.data_loader = DataLoader(input_data)
        self.perturbation_generator = PerturbationGenerator(perturbation_model, num_perturbations, temperature)
        self.response_generator = ModelResponseGenerator(models_dict, instructions, temperature)
        self.similarity_calculator = SimilarityCalculator(similarity_model_name)
        self.keyword_match_calculator = KeywordMatchCalculator()
        self.llm_rating_calculator = LLMRatingCalculator(llm_evaluation_model)

    def run_pipeline(self):
        df = self.data_loader.load_data()
        all_prompts = df["prompt"].unique()  # Get all unique prompts
        perturbations_dict = self.perturbation_generator.get_perturbations_for_all_prompts(all_prompts)
        df_responses = self.response_generator.process_prompts(df, perturbations_dict)
        similarity_results = self.similarity_calculator.calculate_and_aggregate(df_responses)
        keyword_results = self.keyword_match_calculator.calculate_and_aggregate(df_responses)
        rating_results = self.llm_rating_calculator.calculate_and_aggregate(df_responses)
        return similarity_results, keyword_results, rating_results


def del_file():
    temp_files = glob.glob('litellm_*')
    for file in temp_files:
        os.remove(file)


