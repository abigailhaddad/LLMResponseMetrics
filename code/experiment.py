import os
import pandas as pd
import random
import re
from litellm import completion
import os

def set_api_key_for_provider(provider: str):
    """
    Retrieves the API key for the specified provider and sets the
    appropriate environment variable.

    Args:
    provider (str): The provider from which to retrieve the API key.
                    Accepted values are 'OPENAI' or 'ANTHROPIC'.

    Raises:
    ValueError: If the provider is not supported or the API key is not found.
    """
    key_name = f"{provider}_API_KEY"
    api_key = os.getenv(key_name)

    return(api_key)


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
    api_key = set_api_key_for_provider(provider)

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

# Function to aggregate the responses
def aggregate_responses(responses_df: pd.DataFrame):
    # Placeholder for actual aggregation logic
    return responses_df

# Main usage

num_perturbations = 5
models_dict = {'claude-2.1':  "ANTHROPIC", 'gpt-3.5-turbo-0301': "OPENAI",  'gpt-3.5-turbo':  "OPENAI"
               } 
csv_file_path = '../data/prompt_target_answer_pairs.csv'

responses_df = process_prompts(csv_file_path, models_dict, num_perturbations)
aggregated_df = aggregate_responses(responses_df)

# Display the final DataFrame
print(aggregated_df)
