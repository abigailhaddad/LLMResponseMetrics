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
from pathlib import Path

# Define the parameters up top


def read_api_key(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read().strip()

def set_openai_api_key():
    api_key = read_api_key("../keys/openai_key.txt")
    os.environ["OPENAI_API_KEY"] = api_key

def get_random_temperature():
    return random.uniform(0.5, 1.0)  # Adjust the range as needed

def initialize_openai_llm():
    temperature = get_random_temperature()
    return OpenAI(model_name=OPENAI_MODEL, temperature=temperature)

def initialize_similarity_model():
    return SentenceTransformer(TRANSFORMERS_MODEL)

class SimilarJSON(AbstractBehavior):
    def __init__(self, similarity_model: SentenceTransformer, similarity_threshold: float = SIMILARITY_THRESHOLD):
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

def evaluate_row(row):
    json_eval = LLMEval(
        llm=openai_llm,
        expected_behavior=similar_json_behavior,
    )
    return json_eval.evaluate_prompt_correctness(
        pre_context=PRE_CONTEXT,
        prompt=row['prompt'],
        post_context=POST_CONTEXT,
        reference_generation=row['best_answer'],
    )
    
# Assuming the CSV file is in the 'data' folder at the same level as the 'keys' folder
csv_file_path = Path('../data/prompt_best_answer_pairs.csv')
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path).head(2)
# Initialize the models and set the API key
set_openai_api_key()
openai_llm = initialize_openai_llm()
similarity_model = initialize_similarity_model()
similar_json_behavior = SimilarJSON(similarity_model)

df['test_result'] = df.apply(evaluate_row, axis=1)
# Now the DataFrame 'df' contains a new column 'test_result' with the evaluation results
print(df[['prompt', 'best_answer', 'test_result']])
