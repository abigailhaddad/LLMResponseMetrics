# Casual Language Model Evaluation Against Custom Prompts/Answers

The purpose of this is to let users go from a list of LLMs (not just limited to OpenAI) and a set of prompt/answer pairs (questions to ask LLMs and the desired answer) and to see the range of possible answers, including with prompt perturbation (asking semantically-similar questions) and random temperature variance. 

I have three general use cases in mind:

-"How good is this model for the kinds of questions I want to answer?"
-"How robust is this model to different ways of asking a question?"
-"Does the model know the answer to this question/will it ever output the correct answer?"

For this last question, we really just want to look at the best output. For the others, you may want to look at a broader set of responses. This includes code for both returning all of the responses, and for summarizing the prompt/model combinations in terms of the best and worst responses.

We're evaluating "best" and "worst" via similarity scores using a huggingface model with the transformers package. This works well depending on the type of question you're asking and the type of response you expect to get out. Where it doesn't work is if most of the target response and the actual response are just repeating words from the question, because you'll get high similarity scores regardless of whether the actual answer was correct. I have the prompting specifically asking the LLMs for short answers, and that's also how I have my target answers set up, and that's working -- you can see for my responses, the made up questions are consistently returning lower 'best' similarity scores than the non-made up questions.

## Prerequisites

Before you begin, ensure you meet the following requirements:

* You have access to API keys for the LLMs you intend to evaluate. 

## Setup

1. **API Keys**: Place your API keys in the `keys` subfolder, located at the same level as the `code` subfolder. The keys should be in text files named according to the provider, such as `openai_key.txt`.

2. **Data File**: Prepare a CSV file named `prompts.csv` with columns 'prompt' and 'target_answer' that includes the prompts you wish to test and the corresponding expected answers.

## Installation

To install necessary dependencies, run:

```bash
pip install -r requirements.txt


## Using the Toolkit

Launch the demo.ipynb Jupyter notebook to see examples of how to use the toolkit functions to evaluate language models.

## Overview of functions.py

functions.py contains functions for:

Reading API keys from files.
Sending prompts to LLMs.
Parsing the responses from LLMs.
Comparing LLM responses to target answers to evaluate their performance.


## What models can I use?

This is using the litellm library for making API calls to LLMs and uses the transformers library from HuggingFace for the similarity scoring. You can swap out other models which are supported by those libraries, although in both cases just because a model is compatible doesn't mean that it's going to take inputs or return outputs in the way that will work with this code. 

## Contact

For any queries or further discussion, reach out at abigail haddad @ gmail.com

