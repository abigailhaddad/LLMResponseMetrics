import os
import json
import random
from sentence_transformers import SentenceTransformer
from auditor.evaluation.expected_behavior import AbstractBehavior
from langchain.llms import OpenAI
from auditor.evaluation.evaluate import LLMEval
from auditor.utils.similarity import compute_similarity
from typing import List, Tuple, Dict
import pandas as pd

def read_api_key(config):
    file_path = config["OPENAI_API_KEY_PATH"]
    with open(file_path, 'r') as file:
        return file.read().strip()



def get_random_temperature():
    return random.uniform(0.0, 1.0)  # Adjust the range as needed

def initialize_openai_llm(openai_model, config):
    temperature = get_random_temperature()
    api_key = read_api_key(config)
    return OpenAI(model_name=openai_model, temperature=temperature, api_key = api_key)

def initialize_similarity_model(config):
    return SentenceTransformer(config["TRANSFORMERS_MODEL"])

class SimilarJSON(AbstractBehavior):
    def __init__(self, similarity_model: SentenceTransformer, similarity_threshold: float):
        # Note: Do not use SIMILARITY_THRESHOLD directly here. Pass it as an argument when initializing.
        self.similarity_model = similarity_model
        self.similarity_threshold = similarity_threshold
        self.similarity_metric_key = 'Similarity Score'
        self.descriptor = (
            f"Model's generations for perturbations "
            f"are greater than {self.similarity_threshold} similarity metric "
            f"compared to the reference generation AND the answer is in JSON format with the key - answer"
        )

    def is_json_format(self, response: str) -> bool:
        try:
            json_response = json.loads(response)
        except json.JSONDecodeError:
            return False
        return 'answer' in json_response

    def check(self, **kwargs) -> List[Tuple[bool, Dict[str, float]]]:
        test_results = []
        perturbed_generations = kwargs.get('perturbed_generations', [])
        reference_generation = kwargs.get('reference_generation', '')

        for perturbed_gen in perturbed_generations:
            score = compute_similarity(
                sentence_model=self.similarity_model,
                reference_sentence=reference_generation,
                perturbed_sentence=perturbed_gen,
            )
            test_status = score >= self.similarity_threshold and self.is_json_format(perturbed_gen)
            score_dict = {self.similarity_metric_key: round(score, 2)}
            test_results.append((test_status, score_dict))
        return test_results

    def behavior_description(self) -> str:
        return self.descriptor

def evaluate_row(row, openai_model, config):
    openai_llm = initialize_openai_llm(openai_model, config)
    similarity_model = initialize_similarity_model(config)
    similar_json_behavior = SimilarJSON(similarity_model, config["SIMILARITY_THRESHOLD"])

    json_eval = LLMEval(
        llm=openai_llm,
        expected_behavior=similar_json_behavior,
    )
    return json_eval.evaluate_prompt_correctness(
        pre_context=config["PRE_CONTEXT"],
        prompt=row['prompt'],
        post_context=config["POST_CONTEXT"],
        reference_generation=row['target_answer'],
    )


def clean_answer(answer):
    try:
        # Parse the JSON string and return the "answer" value
        return json.loads(answer.strip())['answer']
    except json.JSONDecodeError:
        # If there's an error decoding the JSON, return the original string
        return answer.strip()

def aggregate_results(all_results):
    # Initialize the list that will hold the aggregated data
    aggregated_data = []

    # Iterate over all result dictionaries
    for result_raw in all_results:
        result = result_raw.__dict__
        # Extract necessary data from the current result object's dictionary
        original_prompt = result['original_prompt']
        target_answer = result['reference_generation']
        perturbed_prompts = result['perturbed_prompts']
        perturbed_generations = [clean_answer(ans) for ans in result['perturbed_generations']]
        similarity_scores = [metric['Similarity Score'] for metric in result['metric']]

        # Pair up perturbed prompts with their corresponding cleaned answers
        perturbed_prompts_and_answers = list(zip(perturbed_prompts, perturbed_generations))
        
        # Determine the best and worst answers based on similarity scores
        best_answer_index = similarity_scores.index(max(similarity_scores))
        worst_answer_index = similarity_scores.index(min(similarity_scores))
        best_answer = perturbed_prompts_and_answers[best_answer_index]
        worst_answer = perturbed_prompts_and_answers[worst_answer_index]

        # Append a dictionary for the current prompt to the aggregated data list
        aggregated_data.append({
            'original_prompt': original_prompt,
            'target_answer': target_answer,
            'best_answer_prompt': best_answer[0],
            'best_answer': best_answer[1],
            'worst_answer_prompt': worst_answer[0],
            'worst_answer': worst_answer[1],
        })

    # Convert the aggregated data list to a DataFrame
    return pd.DataFrame(aggregated_data)

def run_evaluation(prompts_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    all_aggregated_results = []

    for model in config["models"]:
        # Set the environment variable for the API key
        # Apply the evaluation to each row
        prompts_df['result'] = prompts_df.apply(
            lambda row: evaluate_row(row, model, config), axis=1
        )
        
        # Aggregate the results and add the model name
        aggregated_results = aggregate_results(prompts_df['result'].tolist())
        aggregated_results['model'] = model
        all_aggregated_results.append(aggregated_results)
    
    # Concatenate all aggregated results into a single DataFrame
    return pd.concat(all_aggregated_results, ignore_index=True)
