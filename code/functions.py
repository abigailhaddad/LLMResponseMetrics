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
        f"Generate a bulleted list of {num_perturbations} sentences "
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
    Loads data from a CSV file or a list.

    Args:
    input_data: File path or list of prompts.
    is_file_path (bool): Flag to indicate if input_data is a file path.

    Returns:
    DataFrame with prompts.
    """
    if is_file_path:
        logging.info("Reading prompts from CSV file.")
        return pd.read_csv(input_data)
    else:
        logging.info("Using direct list of prompts.")
        return pd.DataFrame(input_data, columns=['prompt'])

def process_single_prompt(model, provider, prompt, generate_perturbations, num_perturbations):
    results = []
    perturbations = [prompt] if generate_perturbations else None

    if generate_perturbations:
        logging.info(f"Generating perturbations for prompt: {prompt}")
        perturbations = get_perturbations(prompt, model, provider, num_perturbations)

    for perturbation in perturbations or [prompt]:
        full_query = f"Please answer as briefly as possible: {perturbation if generate_perturbations else prompt}"
        temperature = random.uniform(0.0, 1.0)
        messages = [{"role": "user", "content": full_query}]
        response = call_model(model, messages, provider, temperature)
        generated_text = response['choices'][0]['message']['content']

        result = {
            'model': model,
            'original_prompt': prompt,
            'response': generated_text,
            'temperature': temperature,
        }
        if generate_perturbations:
            result['perturbation'] = perturbation

        results.append(result)

    return results


def perform_similarity_analysis(df, model_name):
    """
    Performs similarity analysis on the DataFrame.

    Args:
    df (DataFrame): DataFrame containing the prompts and responses.
    model_name (str): The HuggingFace model name for encoding texts.

    Returns:
    DataFrame with similarity scores.
    """
    df['similarity_score'] = None
    if 'target_answer' in df.columns:
        logging.info("Performing similarity analysis.")
        for index, row in df.iterrows():
            target_answer = row['target_answer']
            generated_text = row['response']
            similarity_score = calculate_similarity([target_answer], [generated_text], model_name)
            df.at[index, 'similarity_score'] = similarity_score

    return df

def process_prompts(input_data, models_dict, num_perturbations=5, is_file_path=True, generate_perturbations=True, perform_similarity=False):
    """
    Main function to process prompts.

    Args:
    input_data: File path or list of prompts.
    models_dict: Dictionary of models and providers.
    num_perturbations (int): Number of perturbations.
    is_file_path (bool): Flag to indicate if input_data is a file path.
    generate_perturbations (bool): Flag to generate perturbations.
    perform_similarity (bool): Flag to perform similarity analysis.

    Returns:
    DataFrame with results.
    """
    df = load_data(input_data, is_file_path)
    results = []

    for model, provider in models_dict.items():
        for index, row in df.iterrows():
            prompt_results = process_single_prompt(model, provider, row['prompt'], generate_perturbations, num_perturbations)
            results.extend(prompt_results)

    # Create a DataFrame and conditionally add 'perturbation' column
    results_df = pd.DataFrame(results)
    if not generate_perturbations and 'perturbation' in results_df.columns:
        results_df = results_df.drop(columns=['perturbation'])

    if perform_similarity:
        results_df = perform_similarity_analysis(results_df, list(models_dict.keys())[0])  # Assuming model name is key

    logging.info("Processing complete.")
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
    grouped = df_sorted.groupby(['model', 'prompt'])
    best_answers = grouped.head(1).rename(columns={'response': 'best_answer', 'similarity_score': 'best_similarity'})
    worst_answers = grouped.tail(1).rename(columns={'response': 'worst_answer', 'similarity_score': 'worst_similarity'})
    
    # Merge the best and worst answers into one DataFrame
    best_worst_merged = pd.merge(best_answers, worst_answers, on=['model', 'prompt'], suffixes=('_best', '_worst'))
    
    # Select and reorder columns for the final DataFrame
    final_df = best_worst_merged[['model', 'prompt', 'best_answer', 'best_similarity', 'worst_answer', 'worst_similarity']]

    return final_df