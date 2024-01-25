# README for LLM Analysis Pipeline

## Overview

The LLM Analysis Pipeline is a tool which takes existing question/answer pairs and sees how close an LLM can get to producing each target answer. The goal of this is to test if the LLM 'knows' the answer to the question, and to do so in a more automatic and systematic way than you could do using a chat interface and repeatedly asking the same question. 

You can think of this as sampling from a population where the population is the universe of responses that the LLM might give in response to the question. 

There are four general parts to this framework:

1. **Parameters which vary**: The two parameters which vary are initial question, which is perturbed to change the wording while preserving the general meaning, and the temperature. 
2. **Evaluation criteria**: There needs to be a method to evaluate "closeness" to the target answer. This uses three methods described below.
3. **A process for varying parameters**: This tool currently varies both perturbation and temperature randomly. A more sophisticated method would prioritize searching the space around the best-performing parameters.
4. **Stopping criteria**: This currently uses two criteria for stopping the process of getting new responses from the LLM. First, if the maximum ('best') results are stable across a certain number of periods defined by the `stability_threshold` parameter, the process will stop pulling new responses from the LLM. Alternatively, if stability is not reached, the process will stop after a certain number of periods defined by the `max_runs` parameter. 


## Inputs
- Input CSV file with columns: 'prompt', 'target_answer', and 'keywords'. 'Prompt' is the question, 'target_answer' is the answer we're testing closeness to, and 'keywords' are important words appearing in that target answer. You can pass the filepath to the .csv file as a parameter to the pipeline. 
- API keys for the LLM or LLMs being used are stored in a 'keys' directory at the same level as the code. The structure of each API key is `<provider>_key.txt` - for instance, "openai_key.txt".

## Process Flow
1. **Data Preparation**: Loads the necessary data, including prompts and target answers.
2. **Prompt Perturbation**: Generates various versions of each prompt using an LLM.
3. **Response Generation and Evaluation:** Obtains and evaluates responses from the LLM.
4. **Stability/Stopping Criteria:** The pipeline monitors the consistency of maximum evaluation scores across multiple runs, halting further analysis for each prompt once stable maxumum scores are achieved. 
5. **Aggregation:** Aggregates and shows the top responses according to each evaluation method. (This is not part of the pipeline.)


## Evaluation Methods
1. **Keyword Matching**: Analyzes the presence of predefined keywords in the LLM's responses.
2. **Semantic Similarity**: Using a Hugging Face model to put the target answer and LLM's response in the same semantic space so that a silarity scores can be calculated.
3. **LLM Response Analysis**: Uses another LLM to rate how closely the LLM's response matches the target answer.

## Configuration and Sample Parameters
```python
models_dict = {
    'claude-2.1':  "ANTHROPIC", 
    'gpt-3.5-turbo-0301': "OPENAI"
               } # these are the models we're testing
csv_file_path = '../data/prompt_target_answer_pairs.csv'
similarity_model_name = 'sentence-transformers/paraphrase-mpnet-base-v2' # this is the model for calculating similarity scores with
temperature = "variable"
is_file_path = True
max_runs= 10
llm_evaluation_model = ['gpt-4', "OPENAI"]
instructions = "Please answer thoroughly: "
perturbation_model = ['gpt-4', "OPENAI"] # This is the model for generating perturbations. I recommend using GPT-4.
stability_threshold= 3

pipeline = LLMAnalysisPipeline(
    input_data=csv_file_path, 
    models_dict=models_dict, 
    perturbation_model=perturbation_model, 
    llm_evaluation_model=llm_evaluation_model,
    temperature = temperature,
    max_runs= max_runs,
    is_file_path = is_file_path,
    similarity_model_name = similarity_model_name,
    instructions = instructions,
    stability_threshold = stability_threshold

)
```

## Files
- `functions.py`: This has the code for running the pipeline.
- `demo.ipynb`: A Jupyter notebook demonstrating the pipeline usage.
- `population_max_simulation_demo.ipynb`: A Jupyter notebook simulating sampling to find maximum scores in a population.
- `requirements.txt`: Lists all the Python package dependencies.
- `litellm_demo.py`: A simple script showcasing basic usage of the `litellm` library for API calls.

## What Models Can I Use?

- For generating perturbations (the `perturbation_model`) and for testing LLMs (`models_dict`), you can use [chat models supported by LiteLLM](https://docs.litellm.ai/docs/providers). I'd suggest using GPT-4 or a comparably-powered model for perturbation generation because if the model doesn't follow the perturbation instructions, the code won't be able to parse it into a list of perturbations. 
- For generating similarity scores (`similarity_model_name`), you can use any model on HuggingFace with the same model type as 'sentence-transformers/paraphrase-mpnet-base-v2'. 

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
- Set up your .csv file, ensure it has the correct variables, and set the `input_data` parameter to its filepath. 

### Running the Pipeline
1. Import the necessary classes from `functions.py`.
2. Initialize the `LLMAnalysisPipeline` with your specific configurations.
3. Call the `run_pipeline` method to process the data and generate responses.
4. Alternatively, use the `demo.ipynb` notebook.

### Analyzing the Output
The pipeline output DataFrame (`df_responses` if you use `demo.ipynb`) contains the responses along with each metric calculated. You can further analyze these results using your preferred data analysis tools or use the analysis provided in the `demo.ipynb` notebook. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact
If you are using this or want to use this, please get in touch - abigail dot haddad at gmail dot com. 


