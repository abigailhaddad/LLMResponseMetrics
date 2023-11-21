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

def read_api_key(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read().strip()

def set_openai_api_key():
    api_key = read_api_key("../keys/openai_key.txt")
    os.environ["OPENAI_API_KEY"] = api_key

def get_random_temperature():
    return random.uniform(0.5, 1.0)  # Adjust the range as needed

def initialize_openai_llm(openai_model):
    temperature = get_random_temperature()
    return OpenAI(model_name=openai_model, temperature=temperature)

def initialize_similarity_model(transformers_model):
    return SentenceTransformer(transformers_model)

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

def evaluate_row(row, openai_llm, similar_json_behavior, pre_context, post_context):
    json_eval = LLMEval(
        llm=openai_llm,
        expected_behavior=similar_json_behavior,
    )
    return json_eval.evaluate_prompt_correctness(
        pre_context=pre_context,
        prompt=row['prompt'],
        post_context=post_context,
        reference_generation=row['target_answer'],
    )
