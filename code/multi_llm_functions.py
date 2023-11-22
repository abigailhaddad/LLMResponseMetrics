import os
import pandas as pd
import random
import re
from litellm import completion
import os
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

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



# Function to process each prompt in the CSV file
def process_prompts(csv_file_path: str, models_dict: dict, num_perturbations: int = 5):
    df = pd.read_csv(csv_file_path).head(1)
    results = []

    for model, provider in models_dict.items():
        for index, row in df.iterrows():
            original_prompt = row['prompt']
            target_answer = row['target_answer']
            perturbations = get_perturbations(original_prompt, model, provider, num_perturbations)

            for perturbation in perturbations:
                temperature = random.uniform(0.0, 1.0)  # Random temperature for each call
                messages = [{"role": "user", "content": perturbation}]
                response = call_model(model, messages, provider, temperature)
                generated_text = response['choices'][0]['message']['content']
                results.append({
                    'model': model,
                    'original_prompt': original_prompt,
                    'perturbation': perturbation,
                    'response': generated_text,
                    'temperature': temperature,
                    'target_answer':  target_answer,
                })

    return pd.DataFrame(results)

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


