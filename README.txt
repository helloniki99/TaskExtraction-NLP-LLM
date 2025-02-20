*Task Extraction using NLP and LLM Approaches*

This repository contains the code and experiments for my Bachelor's Thesis
"A Comparative Analysis of NLP and LLM Methods for Task Extraction" by Niklas Weiss (Student ID: 12110846).
The thesis investigates several approaches to automated task extraction from process descriptions, comparing traditional NLP methods (e.g., spaCy, Stanza with WordNet, CRF, and BERT-based techniques) with modern large language model (LLM) approaches.

** Project Overview**

The goal of this project is to automate the extraction of tasks from natural language process descriptions, a crucial component in Business Process Management (BPM). The repository includes:

NLP Approaches:

Implementations of traditional task extraction methods using:
* spaCy Dependency Parser
* Stanza in combination with WordNet
* Conditional Random Fields (CRF)
* A BERT-based model for entity classification

LLM Approaches:
Experiments with various large language models using different prompting strategies to extract task-related information from text.

Evaluation:
Scripts for evaluating the extraction performance based on precision, recall, and F1-score. Detailed results and comparative analysis are discussed in the thesis.

** Repository Structure **

./
├── DAT/
│   ├── PET_dataset.csv                # Original PET dataset
│   └── PET_Customized.csv             # Customized dataset 
├── LLM_Testing/
│   ├── Prompt-1/
│   │   ├── Claude/
│   │   │   ├── LLM-Claude3.5-Sonnet.ipynb     # Model: Claude 3.5 Sonnet (claude-3-5-sonnet-20241022)
│   │   │   ├── LLM-Claude3.5-Haiku.ipynb      # Model: Claude 3.5 Haiku (claude-3-5-haiku-20241022)
│   │   │   └── LLM-Claude3-Haiku.ipynb        # Model: Claude 3 Haiku (claude-3-haiku-20240307)
│   │   ├── DeepSeek_OpenAI/
│   │   │   ├── LLM-ChatGpt_4o-mini.ipynb      # Model: GPT-4o-mini (gpt-4o-mini-2024-07-18)
│   │   │   └── LLM-DeepSeek_Chat.ipynb        # Model: DeepSeek-V3 (DeepSeek-V3 2024/12/26)
│   │   ├── Gemini/
│   │   │   ├── LLM-Gemini-1.0-pro.ipynb       # Model: Gemini 1.0 Pro (gemini-1.0-pro)
│   │   │   ├── LLM-Gemini-1.5-Flash.ipynb     # Model: Gemini 1.5 Flash (gemini-1.5-flash-002)
│   │   │   ├── LLM-Gemini-1.5-Flash-8B.ipynb  # Model: Gemini 1.5 Flash-8B (gemini-1.5-flash-8b-001)
│   │   │   └── LLM-Gemini-1.5-Pro.ipynb       # Model: Gemini 1.5 Pro (gemini-1.5-pro-002)
│   │   ├── Grok/
│   │   │   └── LLM_Grok2.ipynb                # Model: Grok2 (grok-2-1212)
│   │   ├── Llama/
│   │   │   ├── LLM-Llama_3.2-8B.ipynb         # Model: Llama-3.1-8B (2024-07-23T17:54:23+00:00)
│   │   │   └── LLM-Llama_70B-Instruct.ipynb   # Model: Llama-3-70B-Instruct (2024-04-18T23:05:24+00:00)
│   │   ├── GPT4o/
│   │   │   └── LLM-GPT4o.ipynb                # Model: GPT-4o (gpt-4o-2024-08-06)
│   │   └── Evaluation/
│   │       └── Eval.ipynb                    # Evaluation for Prompt-1 experiments
│   └── Prompt-2/
│       └── [Same structure and files as Prompt-1, but with the new prompt]
├── NLP_Testing/
    ├── NLP-BERT-Approach.ipynb            # BERT-based task extraction
    ├── NLP-CRF-Approach.ipynb             # CRF-based task extraction
    ├── NLP-spaCy-Approach.ipynb           # spaCy-based task extraction
    └── NLP-StanzaWordNet-Approach.ipynb   # Stanza & WordNet-based extraction

** Requirements **

* torch
* pandas
* spacy
* transformers
* coreferee
* scikit-learn
* sklearn-crfsuite
* numpy
* stanza
* nltk
* anthropic
* openai
* google-generativeai
* requests
* boto3

Notes: 

You will also need to install additional resources for some packages:

For spaCy: en_core_web_sm and optionally en_core_web_trf
For Stanza: English models can be downloaded via stanza.download('en')
For NLTK: Ensure you have the WordNet data (e.g., using nltk.download('wordnet')).

This repository is a work in progress. Updates and refinements will be made as new experiments are conducted and further improvements are developed.

