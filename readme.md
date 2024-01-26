# README for LLM Analysis Pipeline

## Overview

The LLM Analysis Pipeline is a tool for automating asking an LLM a question or set of questions and determining if it 'knows' the answer. That is, if you ask repeatedly and in different ways, what will it say?

There are two different frameworks for this: one where you have just a list of questions (**Open Ended Text Analysis**) and one where you have a list of questions and corresponding answers (**Closed Ended Text Analysis**), and you want to see how close the LLM can get to those answers. 

There are three components to each of these:

1. **Sampling**: We vary parameters (currently, temperature and perturbation, or how we word the question) and we hit the LLM API get to a response. 
2. **Stopping**: We determine when the responses are stable. In the case of **Closed Ended Text Analysis**, that's when when we're not getting any closer to the corresponding text. For **Open Ended Text Analysis**, that's when we're no longer seeing novel responses. At that point, we stop. 
3. **Flagging**: We determine what a person should review. For **Closed Ended Text Analysis**, those are the closest responses. For **Open Ended Text Analysis**, we sample the responses to return a represenative range for review.


## Files
- `requirements.txt`: Lists all the Python package dependencies.
- `ClosedEndedTextAnalysis.ipynb`: A Jupyter notebook demonstrating the pipeline usage for if you have questions and the corresponding answers you're looking for
- `OpenEndedTextAnalysis.ipynb`: A Jupyter notebook demonstrating the pipeline usage for if you have questions but no corresponding answers, and you want to see the full range of outputs
- `functions.py`: This has the functions that are imported in the above .ipynb files.
- `population_max_simulation_demo.ipynb`: A Jupyter notebook simulating sampling to find maximum scores in a population.
- `api_call_demo.py`: A simple script showcasing basic usage of the `litellm` library for API calls.

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact
If you are using this or want to use this, please get in touch - abigail dot haddad at gmail dot com. 


