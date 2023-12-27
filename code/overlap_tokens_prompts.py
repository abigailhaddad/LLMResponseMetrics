import pandas as pd
from openai import OpenAI
import re


def can_be_composed(word, tokens, used_tokens):
    if word == "":
        return True, used_tokens

    for i in range(1, len(word) + 1):
        if word[:i] in tokens:
            success, tokens_used = can_be_composed(
                word[i:], tokens, used_tokens + [word[:i]]
            )
            if success:
                return True, tokens_used

    return False, used_tokens


def load_api_key(file_path):
    """
    Load the OpenAI API key from a file.

    :param file_path: Path to the file containing the API key.
    :return: API key as a string.
    """
    with open(file_path, "r") as file:
        return file.read().strip()


def get_chat_completion(client, model_name, prompt, n=1, temperature=1.0):
    """
    Get n chat completions from the OpenAI API and return a list of responses.

    :param client: OpenAI client object.
    :param model_name: Name of the OpenAI model.
    :param prompt: Prompt to be sent to the model.
    :param n: Number of completions to generate.
    :param temperature: Temperature for the model's response.
    :return: List of tuples containing logprobs content and actual content of the message.
    """
    completions = []
    for _ in range(n):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=5,
            temperature=temperature  # Add temperature parameter
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


def find_words_found_in_both_and_used_tokens(words, tokens):
    """
    Function to find words that are either in tokens or can be composed of tokens,
    along with the tokens used in their composition.
    """
    words_found_in_both = set()
    tokens_used = set()

    for word in words:
        if word in tokens:
            words_found_in_both.add(word)
            tokens_used.add(word)

        success, used_tokens_for_word = can_be_composed(word, tokens, [])
        if success:
            words_found_in_both.add(word)
            tokens_used.update(used_tokens_for_word)

    return words_found_in_both, tokens_used


def find_words_only_in_words(words, tokens):
    words_in_both, _ = find_words_found_in_both_and_used_tokens(words, tokens)
    return words - words_in_both


def find_tokens_not_assigned(all_tokens, words_found_in_both):
    """
    Function to find tokens that are not used as substrings in any word in words_found_in_both.
    """
    tokens_not_assigned = set(all_tokens)
    for word in words_found_in_both:
        for token in all_tokens:
            if token in word:
                tokens_not_assigned.discard(token)

    return tokens_not_assigned


def analyze_responses_vs_logits(client, model_name, prompt, n, temperature):
    completions = get_chat_completion(client, model_name, prompt, n, temperature)

    all_tokens = set()
    all_words = set()
    all_logits = []

    for logprobs_content, response in completions:
        split_response = re.split(r"\s+|[\.,;:\-!?]", response.lower())
        words = set([re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", i) for i in split_response])

        all_tokens.update(set(extract_tokens(process_logprobs(logprobs_content))))
        all_words.update(words)
        all_logits.append(logprobs_content)

    words_in_both, _ = find_words_found_in_both_and_used_tokens(all_words, all_tokens)
    words_only_in_words = find_words_only_in_words(all_words, all_tokens)
    tokens_not_used = find_tokens_not_assigned(all_tokens, words_in_both)

    return {
        "Responses": [response for _, response in completions],
        "ResponseTokens": [response_tokens for response_tokens, _ in completions],
        "WordsFoundOnlyInResponses": sorted(list(words_only_in_words)),
        "WordsFoundInBoth": sorted(list(words_in_both)),
        "LogitsNotAssignedToAnyWords": sorted(list(tokens_not_used)),
        "Logits": all_logits,
    }


def are_logits_same(logits_list):
    if not logits_list:
        return True

    first_logits = logits_list[0]
    return all(logits == first_logits for logits in logits_list[1:])

def count_logits_in_responses(logits_list, n):
    """
    Count how many logits appear in a specific number of responses.

    :param logits_list: List of logits for each response.
    :param n: Total number of responses.
    :return: Dictionary with keys as number of responses and values as count of logits.
    """
    logit_frequency = {}
    for response_logits in logits_list:
        for token_logprob in response_logits:
            token = token_logprob.token  # Accessing the token string
            logit_frequency[token] = logit_frequency.get(token, 0) + 1

    frequency_count = {i: 0 for i in range(1, n+1)}
    for freq in logit_frequency.values():
        if freq in frequency_count:
            frequency_count[freq] += 1

    return frequency_count





key_file_path = "keys/openai_key.txt"
api_key = load_api_key(key_file_path)

client = OpenAI(api_key=api_key)

model_name = "gpt-3.5-turbo"
prompt = "What is the process of photosynthesis?"
n = 3  # Number of completions for each prompt
# Define temperatures to analyze
temperatures = [0.2, 0.5, 0.7, 1.0]
results_list = []

for temp in temperatures:
    results = analyze_responses_vs_logits(client, model_name, prompt, n, temperature=temp)
    logits_count = count_logits_in_responses(results["Logits"], n)
    results_list.append(pd.DataFrame({'Temperature': [temp], 'LogitCounts': [logits_count]}))

# Concatenate all results into a single DataFrame
df = pd.concat(results_list, ignore_index=True)

print(df)
