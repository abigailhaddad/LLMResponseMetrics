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


def del_file():
    temp_files = glob.glob('litellm_*')
    for file in temp_files:
        os.remove(file)


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


# Function to parse the model response
def parse_model_response(response):
    content = response['choices'][0]['message']['content']
    if content.startswith('1.'):
        perturbations = re.split(r'\n\d+\.\s*', content)
        perturbations = [pert.strip() for pert in perturbations if pert.strip()]
    else:
        perturbations = content.split('\n- ')[1:]
        perturbations = [pert.strip('* ') for pert in perturbations]
    return perturbations

# Function to get paraphrased sentences (perturbations) from the model
def get_perturbations(prompt: str, model: str, provider: str, num_perturbations: int = 5, temperature: float = 0.5):
    paraphrase_instruction = (
        f"Generate a bulleted list of {num_perturbations + 1} sentences " # I don't know why this is necessary but I'm consisting getting off by 1 when I don't include it
        f"with the same meaning as \"{prompt}\""
    )
    messages = [{"role": "user", "content": paraphrase_instruction}]
    response = call_model(model, messages, provider, temperature)
    perturbations = parse_model_response(response)
    return perturbations


def call_model(model: str, messages: list, provider: str, temperature: float):
    """
    Calls the specified model using the provider's API key and the given parameters.

    Args:
    model (str): Model identifier.
    messages (list): List of message dictionaries to send.
    provider (str): The provider of the API ('OPENAI' or 'ANTHROPIC').
    temperature (float): The temperature setting for the language model.

    Returns:
    The API response.
    """
    # Set the API key for the provider
    api_key = read_api_key(provider)
    # Call the completion endpoint with the provided parameters
    response = completion(model=model, messages=messages, temperature=temperature, api_key = api_key)
    return response


def load_data(input_data, is_file_path=True):
    """
    Loads data from a CSV file or a list and processes the 'keywords' column.

    Args:
    input_data: File path or list of prompts.
    is_file_path (bool): Flag to indicate if input_data is a file path.

    Returns:
    DataFrame with prompts and processed keywords.
    """
    if is_file_path:
        logging.info("Reading prompts from CSV file.")
        df = pd.read_csv(input_data, encoding='ISO-8859-1')
    else:
        logging.info("Using direct list of prompts.")
        df = pd.DataFrame(input_data, columns=['prompt'])

    if 'keywords' in df.columns:
        df['keywords'] = df['keywords'].apply(lambda x: [keyword.strip().lower() for keyword in x.split('\n') if keyword.strip()])
    
    return df



def process_single_prompt(model, provider, prompt, num_perturbations, instructions, temperature, target_answer=None, keywords=None):
    perturbations = [prompt] + (get_perturbations(prompt, model, provider, num_perturbations) if num_perturbations else [])
    temp_func = lambda: random.uniform(0.0, 1.0) if temperature == "variable" else temperature

    results = [
        create_result_dict(
            model, prompt, perturbation, call_model(
                model, [{"role": "user", "content": f"{instructions} {perturbation}"}], provider, temp_func()
            ), temp_func(), target_answer, keywords
        )
        for perturbation in perturbations
    ]

    return results

def create_result_dict(model, original_prompt, actual_prompt, response, temperature, target_answer, keywords):
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

def perform_similarity_analysis(df, model_name):
    if 'target_answer' in df.columns:
        logging.info("Performing similarity analysis.")
        df['similarity_score'] = df.apply(
            lambda row: calculate_similarity([row['target_answer']], [row['response']], model_name) if pd.notnull(row['target_answer']) else None, 
            axis=1
        )
    return df

def process_prompts(input_data, models_dict, num_perturbations=0, num_runs=1, is_file_path=True, perform_similarity=False, instructions="Please answer as briefly as possible: ", temperature=0.7):
    df = load_data(input_data, is_file_path)
    results = []

    for model, provider in models_dict.items():
        process_model_prompts(model, provider, df, num_perturbations, num_runs, results, instructions, temperature)

    results_df = create_results_df(results, num_perturbations, perform_similarity, models_dict)
    logging.info("Processing complete.")
    return results_df

def process_model_prompts(model, provider, df, num_perturbations, num_runs, results, instructions, temperature):
    for index, row in df.iterrows():
        target_answer = row['target_answer'] if 'target_answer' in df.columns else None
        keywords = row['keywords'] if 'keywords' in df.columns else None
        process_row(model, provider, row['prompt'], num_perturbations, num_runs, results, instructions, temperature, target_answer, keywords)

def process_row(model, provider, prompt, num_perturbations, num_runs, results, instructions, temperature, target_answer, keywords):
    for _ in range(num_runs):
        prompt_results = process_single_prompt(model, provider, prompt, num_perturbations, instructions, temperature, target_answer, keywords)
        results.extend(prompt_results)

def create_results_df(results, num_perturbations, perform_similarity, models_dict):
    results_df = pd.DataFrame(results)
    if not num_perturbations and 'actual_prompt' in results_df.columns:
        results_df.drop(columns=['actual_prompt'], inplace=True)

    if perform_similarity:
        results_df = perform_similarity_analysis(results_df, list(models_dict.keys())[0])  # Assuming model name is key

    return results_df

def get_model(model_name: str):
    """
    Loads a HuggingFace model and tokenizer based on the model name.

    Args:
    model_name (str): The HuggingFace model name.

    Returns:
    model, tokenizer: The loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def encode_texts(texts: list, model, tokenizer):
    """
    Encodes a list of texts into embeddings using the provided model and tokenizer.

    Args:
    texts (list): A list of strings to encode.
    model: The HuggingFace model.
    tokenizer: The HuggingFace tokenizer.

    Returns:
    embeddings: The encoded embeddings for the texts.
    """
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings

def calculate_similarity(target_texts: list, actual_texts: list, model_name: str):
    """
    Calculates the similarity scores between target and actual texts.

    Args:
    target_texts (list): The target outputs.
    actual_texts (list): The actual outputs.
    model_name (str): The HuggingFace model name for encoding texts.

    Returns:
    similarities (list): The list of similarity scores.
    """
    model, tokenizer = get_model(model_name)
    target_embeddings = encode_texts(target_texts, model, tokenizer)
    actual_embeddings = encode_texts(actual_texts, model, tokenizer)
    
    # Calculate cosine similarities
    similarities = [1 - cosine(target_embedding, actual_embedding)
                    for target_embedding, actual_embedding in zip(target_embeddings, actual_embeddings)]
    
    return similarities


def aggregate_results(df):
    # Sort the DataFrame based on similarity scores
    df_sorted = df.sort_values(by='similarity_score', ascending=False)

    # Group by 'model' and 'prompt' and get the first (best) and last (worst) after sorting
    grouped = df_sorted.groupby(['model', 'original_prompt'])
    best_answers = grouped.head(1).rename(columns={'response': 'best_answer', 'similarity_score': 'best_similarity'})
    worst_answers = grouped.tail(1).rename(columns={'response': 'worst_answer', 'similarity_score': 'worst_similarity'})
    
    # Merge the best and worst answers into one DataFrame
    best_worst_merged = pd.merge(best_answers, worst_answers, on=['model', 'original_prompt'], suffixes=('_best', '_worst'))
    
    # Select and reorder columns for the final DataFrame
    final_df = best_worst_merged[['model', 'original_prompt', 'best_answer', 'best_similarity', 'worst_answer', 'worst_similarity']]

    return final_df

def calculate_keyword_match_percent(target_keywords: list, actual_responses: list):
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

def add_keyword_match_percentages(df, target_keywords_column='keywords'):
    df['keyword_match_percent'] = df.apply(
            lambda row: calculate_keyword_match_percent(row[target_keywords_column], [row['response']])[0]
            if row[target_keywords_column] else None, 
            axis=1
        )
    return df

    
def aggregate_keyword_results(df):
    # Sort the DataFrame based on keyword match percent
    df_sorted = df.sort_values(by='keyword_match_percent', ascending=False)

    # Group by 'model' and 'original_prompt' and get the best and worst keyword matches
    grouped = df_sorted.groupby(['model', 'original_prompt'])
    best_keyword_matches = grouped.head(1).rename(columns={'response': 'best_keyword_match', 'keyword_match_percent': 'best_keyword_match_percent'})
    worst_keyword_matches = grouped.tail(1).rename(columns={'response': 'worst_keyword_match', 'keyword_match_percent': 'worst_keyword_match_percent'})

    # Merge the best and worst keyword matches into one DataFrame
    best_worst_merged = pd.merge(best_keyword_matches, worst_keyword_matches, on=['model', 'original_prompt'], suffixes=('_best', '_worst'))

    # Select and reorder columns for the final DataFrame
    final_df = best_worst_merged[['model', 'original_prompt', 'best_keyword_match', 'best_keyword_match_percent', 'worst_keyword_match', 'worst_keyword_match_percent']]

    return final_df



def rate_response_with_llm(row, llm_evaluation_model):
    model, provider = llm_evaluation_model
    target_answer = row['target_answer']
    actual_response = row['response']

    # Formulate the prompt for the LLM
    rating_prompt = f"Rate the following response on an integer scale from 1 to 10 based on its similarity to the target answer. Only return an integer, with no comments or punctuation \n\nTarget Answer: {target_answer}\nResponse: {actual_response}\nRating:"

    # Call the model
    response = call_model(model, [{"role": "user", "content": rating_prompt}], provider, temperature=0.7)
    rating = response['choices'][0]['message']['content'].strip()

    # Extract and return the rating number
    try:
        return int(rating)
    except ValueError:
        logging.warning(f"Could not extract a valid rating from response: {rating}")
        return None

def add_ratings_to_df(df, llm_evaluation_model):
    df['rating'] = df.apply(lambda row: rate_response_with_llm(row, llm_evaluation_model), axis=1)
    return df


def aggregate_rating_results(df):
    # Shuffle the DataFrame to randomly break ties
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    # Sort the DataFrame based on rating
    df_sorted = df_shuffled.sort_values(by='rating', ascending=False)

    # Group and get the best and worst rated matches
    grouped = df_sorted.groupby(['model', 'original_prompt'])
    best_rated = grouped.head(1).rename(columns={'response': 'best_rated', 'rating': 'best_rating'})
    worst_rated = grouped.tail(1).rename(columns={'response': 'worst_rated', 'rating': 'worst_rating'})

    # Merge the best and worst rated into one DataFrame
    best_worst_merged = pd.merge(best_rated, worst_rated, on=['model', 'original_prompt'], suffixes=('_best', '_worst'))

    final_df = best_worst_merged[['model', 'original_prompt', 'best_rated', 'best_rating', 'worst_rated', 'worst_rating']]

    return final_df
