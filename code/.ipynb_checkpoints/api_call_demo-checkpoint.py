
import litellm
import pandas as pd

def get_response(model, prompt, temperature):
    messages = [{"content": prompt, "role": "user"}]
    response = litellm.completion(model=model, messages=messages, temperature=temperature)
    return response

def generate_responses(model, prompts, temperature, num_repeats):
    responses_data = [
        {
            "model": model,
            "prompt": prompt,
            "response": get_response(model, prompt, temperature)['choices'][0]['message']['content'],
            "temperature": temperature,
            "repeat": repeat + 1
        }
        for prompt in prompts for repeat in range(num_repeats)
    ]

    responses_df = pd.DataFrame(responses_data)
    return responses_df

# Example usage
prompts = ["Hello, how are you?", "What is the weather like today?"]
model = "gpt-3.5-turbo"
responses_df = generate_responses(model, prompts, temperature=0.7, num_repeats=2)
