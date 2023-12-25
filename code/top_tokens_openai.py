from openai import OpenAI


def load_api_key(file_path):
    """
    Load the OpenAI API key from a file.

    :param file_path: Path to the file containing the API key.
    :return: API key as a string.
    """
    with open(file_path, "r") as file:
        return file.read().strip()


def get_chat_completion(client, model_name, prompt):
    """
    Get the chat completion from the OpenAI API.

    :param client: OpenAI client object.
    :param model_name: Name of the OpenAI model.
    :param prompt: Prompt to be sent to the model.
    :return: Logprobs content from the model's response.
    """
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=5,
    )
    return completion.choices[0].logprobs.content


def process_logprobs(logprobs_content):
    """
    Process log probabilities to extract token probabilities.

    :param logprobs_content: Logprobs content from the model's response.
    :return: List of sorted token probabilities.
    """
    token_probs = [
        (top_logprob.token, top_logprob.logprob)
        for token_logprob in logprobs_content
        for top_logprob in token_logprob.top_logprobs
    ]
    return sorted(token_probs, key=lambda x: x[1], reverse=True)


def keyword_analysis(tokens, keywords):
    """
    Analyze which keywords can be formed from the tokens, considering non-sequential combinations.

    :param tokens: List of tokens.
    :param keywords: List of keywords to search for.
    :return: Dictionary with keys True and False, values are lists of found and not found keywords.
    """
    found = set()

    for keyword in keywords:
        relevant_tokens = [token for token in tokens if token in keyword]
        combined = "".join(relevant_tokens)
        if combined == keyword:
            found.add(keyword)

    not_found = set(keywords) - found
    return {True: list(found), False: list(not_found)}


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
        if processed_token not in seen:
            seen.add(processed_token)
            tokens.append(processed_token)
    return tokens


def keyword_analysis(tokens, keywords):
    """
    Analyze which keywords can be formed from the tokens, using backtracking to handle multiple paths.

    :param tokens: List of tokens.
    :param keywords: List of keywords to search for.
    :return: Dictionary with keys True and False, values are lists of found and not found keywords.
    """
    found = set()

    def can_form_keyword(keyword, token_list):
        if not keyword:
            return True
        for i, token in enumerate(token_list):
            if keyword.startswith(token):
                if can_form_keyword(keyword[len(token) :], token_list[i + 1 :]):
                    return True
        return False

    for keyword in keywords:
        if can_form_keyword(keyword, tokens):
            found.add(keyword)

    not_found = set(keywords) - found
    return {True: list(found), False: list(not_found)}


def main():
    key_file_path = "keys/openai_key.txt"
    api_key = load_api_key(key_file_path)

    client = OpenAI(api_key=api_key)

    model_name = "gpt-3.5-turbo"
    prompt = "What is the process of photosynthesis?"
    keywords = ["chlorophyll", "sunlight", "mitochondria"]

    logprobs_content = get_chat_completion(client, model_name, prompt)
    sorted_token_probs = process_logprobs(logprobs_content)
    sorted_tokens = extract_tokens(sorted_token_probs)

    keyword_results = keyword_analysis(sorted_tokens, keywords)
    return (sorted_tokens, keyword_results)


if __name__ == "__main__":
    sorted_tokens, keyword_results = main()


def run_tests():
    test_cases = [
        # Test Case 1: Simple Match
        (
            ["chlorophyll", "sunlight", "mitochondria"],
            ["chlorophyll", "sunlight", "mitochondria"],
            {True: ["chlorophyll", "sunlight", "mitochondria"], False: []},
        ),
        # Test Case 2: Sequential Split
        (
            ["chloro", "phyll", "sun", "light", "mito", "chondria"],
            ["chlorophyll", "sunlight", "mitochondria"],
            {True: ["chlorophyll", "sunlight", "mitochondria"], False: []},
        ),
        # Test Case 3: Non-Sequential Split
        (
            ["chloro", "sun", "phyll", "light", "mito", "chondria"],
            ["chlorophyll", "sunlight", "mitochondria"],
            {True: ["chlorophyll", "sunlight", "mitochondria"], False: []},
        ),
        # Test Case 4: Multiple Possible Paths
        (
            ["chloro", "phyl", "phyll", "light"],
            ["chlorophyll"],
            {True: ["chlorophyll"], False: []},
        ),
        # Test Case 5: No Match
        (
            ["water", "carbon", "dioxide"],
            ["chlorophyll", "sunlight", "mitochondria"],
            {True: [], False: ["chlorophyll", "sunlight", "mitochondria"]},
        ),
    ]

    for i, (tokens, keywords, expected) in enumerate(test_cases, 1):
        result = keyword_analysis(tokens, keywords)
        result = {
            k: sorted(v) for k, v in result.items()
        }  # Sort the lists for consistent comparison
        expected = {k: sorted(v) for k, v in expected.items()}
        assert (
            result == expected
        ), f"Test Case {i} Failed: Expected {expected}, got {result}"
        print(f"Test Case {i} Passed")


run_tests()
