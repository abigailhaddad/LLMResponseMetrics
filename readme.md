# README for LLM Analysis Pipeline

## Overview

The LLM Analysis Pipeline is a tool designed to evaluate different methods for testing whether an LLM 'knows the answer' to a given question. This is done via having an initial set of questions and their target answers, prompting an LLM to answer the same question (but asked in different ways), and then taking the best answers to each question from each model, given three different evaluation methods.

Some of the questions in the input data are 'true', meaning that the we already know the LLM knows the answer and some are 'false', meaning that it does not. But if that's not how you want to use this, the pipeline will still run without that, you just won't be able to make a graph in the demo.ipynb workbook. 


## Core Components and Evaluation Methods
1. **Keyword Matching**: Analyzes the presence of predefined keywords in the LLM's responses.
2. **Semantic Similarity**: Using Hugging Face's sentence-transformers to calculate similarity scores between LLM responses and target answers.
3. **LLM Response Analysis**: Uses another LLM to rate how closely the LLM's response matches the target answer.

## Data Input Structure
- Input CSV file with columns: 'prompt', 'target_answer', 'keywords', and 'true_or_false'.
- 'true_or_false': Indicates whether the LLM is expected to know the answer ('true') or not ('false').
- 'keywords': Specific keywords related to each prompt for the keyword matching method.
- API keys for the LLMs stored in a 'keys' directory.

## Process Flow
1. **Data Preparation**: Loads prompts, target answers, and associated data including the true/false flag.
2. **Prompt Perturbation**: Generates variations or perturbations of each prompt using an LLM.
3. **Response Generation**: Obtains responses from an LLM for both original and perturbed prompts.
4. **Evaluation**: Applies the three methods to assess each response.
5. **Aggregation**: Analyzes the best outputs according to each method.

## Configuration and Sample Parameters
```python
models_dict = {
    'gpt-3.5-turbo-0301': "OPENAI"
}
csv_file_path = '../data/prompt_target_answer_pairs.csv'
similarity_model_name = 'sentence-transformers/paraphrase-mpnet-base-v2'
num_runs = 1
temperature = .9
num_perturbations= 0
is_file_path = True
llm_evaluation_model = ['gpt-4', "OPENAI"]
instructions = "Please answer thoroughly: "
perturbation_model = ['gpt-4', "OPENAI"]

pipeline = LLMAnalysisPipeline(
    input_data=csv_file_path, 
    models_dict=models_dict, 
    perturbation_model=perturbation_model, 
    llm_evaluation_model=llm_evaluation_model,
    temperature = temperature,
    num_runs= num_runs,
    is_file_path = is_file_path,
    similarity_model_name = similarity_model_name,
    num_perturbations = num_perturbations,
    instructions = instructions
)
```

## Other Files
- `demo.ipynb`: A Jupyter notebook demonstrating the pipeline usage.
- `requirements.txt`: Lists all the Python package dependencies.
- `litellm_demo.py`: A simple script showcasing basic usage of the `litellm` library for API calls.
- `keys/`: Directory for storing API keys.



# README for LLM Analysis Pipeline

## Overview
The LLM (Large Language Model) Analysis Pipeline is a Python-based toolkit designed to process, perturb, and evaluate language model outputs. It's tailored for researchers and developers working with language models like GPT-3, GPT-4, or similar, providing tools for generating perturbed prompts, calling language models, and evaluating their outputs using various metrics.

## Features
- **Data Loading**: Load prompts from CSV files or direct lists.
- **API Key Handling**: Securely read API keys for different language model providers.
- **Perturbation Generation**: Create perturbations of prompts to test language model robustness.
- **Response Generation**: Generate responses from various language models.
- **Result Aggregation**: Aggregate results based on specific metrics.
- **Similarity Scoring**: Calculate similarity scores between target and actual responses.
- **Keyword Matching**: Perform keyword analysis on the responses.
- **Rating Calculation**: Rate responses based on their similarity to target answers.

## Installation
1. Clone the repository to your local machine.
2. Install the required packages using `pip`:
   ```
   pip install -r requirements.txt
   ```
3. Create a `keys` folder in the root directory and add your API keys in the format `<provider>_key.txt`.

## Usage

### Setting Up
- Ensure that your API keys are correctly placed in the `keys` folder.
- Place your data in the `data` folder if you're using CSV files for input.

### Running the Pipeline
1. Import the necessary classes from `functions.py`.
2. Initialize the `LLMAnalysisPipeline` with your specific configurations.
3. Call the `run_pipeline` method to process the data and generate responses or use the demo.ipynb notebook.


### Analyzing the Output
The output DataFrame `df_responses` contains the responses along with various metrics calculated. You can further analyze these results using your preferred data analysis tools.

## Contributing
If you are using this or want to use this, please get in touch - abigail dot haddad at gmail dot com. 

