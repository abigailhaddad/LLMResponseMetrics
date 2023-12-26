import pandas as pd
from openai import OpenAI
import re


def load_api_key(file_path):
    """
    Load the OpenAI API key from a file.

    :param file_path: Path to the file containing the API key.
    :return: API key as a string.
    """
    with open(file_path, "r") as file:
        return file.read().strip()


def get_chat_completion(client, model_name, prompt, n=1):
    """
    Get n chat completions from the OpenAI API and return a list of responses.

    :param client: OpenAI client object.
    :param model_name: Name of the OpenAI model.
    :param prompt: Prompt to be sent to the model.
    :param n: Number of completions to generate.
    :return: List of tuples containing logprobs content and actual content of the message.
    """
    completions = []
    for _ in range(n):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=5,
        )
        logprobs_content = completion.choices[0].logprobs.content
        actual_content = completion.choices[0].message.content
        completions.append((logprobs_content, actual_content))
    return completions


def process_logprobs(logprobs_content):
    """
    Process log probabilities to extract token probabilities.

    :param logprobs_content: Logprobs content from the model's response.
    :return: List of sorted token probabilities.
    """
    token_probs = [
        (token_logprob.token, token_logprob.logprob)
        for token_logprob in logprobs_content
    ]
    return sorted(token_probs, key=lambda x: x[1], reverse=True)


def extract_tokens(sorted_token_probs):
    """
    Extract tokens from sorted token probabilities, convert to lowercase, strip, and remove duplicates.

    :param sorted_token_probs: List of sorted token probabilities.
    :return: List of unique, processed tokens.
    """
    seen = set()
    tokens = []
    for token, _ in sorted_token_probs:
        processed_token = token.lower().strip()
        # Use regular expression to strip non-alphabetical characters from both ends
        processed_token = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", processed_token)

        if processed_token not in seen:
            seen.add(processed_token)
            tokens.append(processed_token)
    return tokens


def analyze_responses_vs_logits(client, model_name, prompt, n):
    completions = get_chat_completion(client, model_name, prompt, n)
    
    words_found_only_in_responses = set()
    words_found_in_both = set()
    logits_not_assigned_to_any_words = set()
    
    for logprobs_content, response in completions:
        response_tokens = extract_tokens(process_logprobs(logprobs_content))
        
        response_tokens_set = set([re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", i.lower() ) for i in response.split()])
        intersection = response_tokens_set.intersection(response_tokens)
        words_found_in_both.update(intersection)
        words_found_only_in_responses.update(response_tokens_set - intersection)
        
        # Convert response_tokens to a set before performing the set difference operation
        logits_not_assigned_to_any_words.update(set(response_tokens) - intersection)
    
    return {
        "Responses": [response for _, response in completions],
        "ResponseTokens": [response_tokens for response_tokens, _ in completions],
        "WordsFoundOnlyInResponses": sorted(list(words_found_only_in_responses)),
        "WordsFoundInBoth": sorted(list(words_found_in_both)),
        "LogitsNotAssignedToAnyWords": sorted(list(logits_not_assigned_to_any_words))
    }


def main():
    key_file_path = "keys/openai_key.txt"
    api_key = load_api_key(key_file_path)

    client = OpenAI(api_key=api_key)

    model_name = "gpt-3.5-turbo"
    prompt = "What is the process of photosynthesis?"
    n = 3  # Number of completions for each prompt

    # Call specific pieces and return objects
    results = analyze_responses_vs_logits(client, model_name, prompt, n)
    
    print("Responses:")
    print(results["Responses"])
    print("\nResponse Tokens:")
    print(results["ResponseTokens"])
    return results


if __name__ == "__main__":
    main_result = main()
