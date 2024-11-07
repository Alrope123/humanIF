# HREF: Human Reference Guided Evaluation for Instruction Following

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

[📑 Paper]() | [🤗 Leaderboard]() | [🤗 Validation Set]() | [🤗 Human Agreement Set]()
 

### Announcement
* **11/07/2024**: We officially publish the paper [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251), along with this codebase, the [HREF leaderboard](), [the validaiton set](), and [the human agreement set]()! 

### Citation
```

```

## Install
Make a new Python 3.10 environment and install the `href` package.

```bash
conda create -n href python=3.10
conda activate href
pip install -e .
```

## Evaluation on the Validation Set
To evaluate a open huggingface or local model, run:
```bash
href evaluate --model_name_or_path meta-llama/Llama-3.1-8B-Instruct
```
<details>
<summary> General arguments </summary>

- `--model_name_or_path`: the huggingface model name or the path to a local directory that contains the model to use for evaluation.
- `--dataset`: the huggingface dataset name or the path to a local file to use for evaluation. Default to use the validation set of HREF.
- `--split`: the split to use in `dataset`.
- `--nr_cateogry`: categories in the HREF to include. Default to use all 8 categories.
- `--seed`: random seed.
- `--save_dir`: directory to save all results.
- `--cache_dir`: the directory to store downloaded datasets, models, and intermmediate annotation files
</details>

<details>
<summary> Evaluation arguments </summary>

- `annotator`: name of the evaluation methods. It has to be one the three following: 1. a basic annotator defined in `evaluation/evaluators.DEFINED_ANNOTATORS`. 2. a configuration name for LLM-as-a-Judge that corresponds to a directory in `llm-as-a-judge`. 3. a suite of the above two types of unit evaluators defined in `evaluation/evaluators.DEFINED_ANNOTATOR_SUITE_DICT`. Default to be suite `ahref` that we defined in our paper.
- `--config_dir`: the directory to contain configures for LLM-as-a-Judge evaluators.
- `--use_human_reference`: whether of not `annotator` needs to use the human reference. No need to specify if `annotator` specifies a evaluator suite. 
</details>

<details>
<summary> Generation arguments </summary>

- `--response_dir`: the directory that contains pre-generated model outputs. If specified, we will skip output generation and jump directly into evaluation.
- `--use_vllm`: if given, we will use vLLM to generate the responses.
- `--tokenizer_name_or_path`: the huggingface tokenizer name or the path to a local directory that contains the tokenizer to use for evaluation. If not specified, we will use the same ones as `model_name_or_path`.
- `--use_slow_tokenizer`: if given, we will use the slow tokenizer.
- `--max_new_tokens`: maximum number of new tokens to generate.
- `--temperature`: the temperature we use for model generation. Default to be 0.0.
- `--batch_size`: batch size for generation.
- `--load_in_8bit`: load model in 8bit mode, which will reduce memory and speed up inference.
- `--gptq`: if given, we're evaluating a 4-bit quantized GPTQ model.
- `--use_chat_format`: if given, we will use the chat format for the prompts.
- `--chat_formatting_function`: the name of the function to use to create the chat format. This function will be dynamically imported. Functions are specified in `generation/templates.py`. Default to use the chat template in the tokenizer.
</details>

<br>

To evaluate an OpenAI models, run:
```bash
OPENAI_API_KEY=<your OpenAI key>
href evaluate --model_name_or_path gpt-4
```
<details>
<summary> General arguments </summary>

- `--model_name_or_path`: the huggingface model name or the path to a local directory that contains the model to use for evaluation.
- `--dataset`: the huggingface dataset name or the path to a local file to use for evaluation. Default to use the validation set of HREF.
- `--split`: the split to use in `dataset`.
- `--nr_cateogry`: categories in the HREF to include. Default to use all 8 categories.
- `--seed`: random seed.
- `--save_dir`: directory to save all results.
- `--cache_dir`: the directory to store downloaded datasets, models, and intermmediate annotation files
</details>

<details>
<summary> Evaluation arguments </summary>

- `annotator`: name of the evaluation methods. It has to be one the three following: 1. a basic annotator defined in `evaluation/evaluators.DEFINED_ANNOTATORS`. 2. a configuration name for LLM-as-a-Judge that corresponds to a directory in `llm-as-a-judge`. 3. a suite of the above two types of unit evaluators defined in `evaluation/evaluators.DEFINED_ANNOTATOR_SUITE_DICT`. Default to be suite `ahref` that we defined in our paper.
- `--config_dir`: the directory to contain configures for LLM-as-a-Judge evaluators.
- `--use_human_reference`: whether of not `annotator` needs to use the human reference. No need to specify if `annotator` specifies a evaluator suite. 
</details>

<details>
<summary> Generation arguments </summary>

- `--response_dir`: the directory that contains pre-generated model outputs. If specified, we will skip output generation and jump directly into evaluation.
- `--max_new_tokens`: maximum number of new tokens to generate.
- `--temperature`: the temperature we use for model generation. Default to be 0.0. 
</details>

### Evaluate your own model
There are three ways to evaluate your own model/tokenizer on the validation set:

#### 1. Adapt transformers interface
1. Adapt your model/tokenizer to the [transfomers interface](https://huggingface.co/docs/transformers/index)
2. Either upload your model/tokenizer on huggingface, or save them locally.
3. Specify the `--model_name_or_path` argument with either the name of your uploaded model/tokenizer on huggingface, or the path to your local directory that contains saved model/tokenizer.
4. If you would like to use a customized chat format on the instructions when generating the model response, add your own function in `href/generation/templates.py` and pass its path as an argument like `--chat_formatting_function href.generation.templates.your_function`.

#### 2. Pre-generate responses
1. Generate model responses to the `instruction` field of the data in your own way.
2. Make sure to save responses in the following struction 
```
📂 <your save directory>
 ┣ 📂 Brainstorm
 ┃ ┗ 📜 responses.jsonl
 ┣ 📂 Open QA
 ┃ ┗ 📜 responses.jsonl
 ┣ ...
 ┃ ...
 ┣ 📂 Multi-Document Sythesis
 ┃ ┗ 📜 responses.jsonl
```
where each data point in responses.jsonl contains the fields: `instruction`, `output`, `generator`.

3. Pass the save directory like `--response_dir <your save directory>`. Note that you should still pass a custom name for the model with `--model_name_or_path`.

#### 3. Add more API models
Please follow the logic how we implement OpenAI API in `href/generation/generate.py` to add your own API model, which is relatively straightforward.


## Human Agreement Analysis
To calculate the human agreement rate of an evaluation method on HREF human agreement set, run:
```bash
href calculate_agreement --annotator meta-llama/Llama-3.1-8B-Instruct
```
<details>
<summary> General arguments </summary>

- `--dataset`: the huggingface dataset name or the path to a local file to use for analysis. Default to use the human agreement set of HREF.
- `--split`: the split to use in `dataset`.
- `--nr_cateogry`: categories in the HREF to include. Default to use all 8 categories.
- `--seed`: random seed.
- `--save_dir`: directory to save all results.
- `--cache_dir`: the directory to store downloaded datasets, models, and intermmediate annotation files
</details>

<details>
<summary> Evaluation arguments </summary>

- `annotator`: name of the evaluation methods. It has to be one the three following: 1. a basic annotator defined in `evaluation/evaluators.DEFINED_ANNOTATORS`. 2. a configuration name for LLM-as-a-Judge that corresponds to a directory in `llm-as-a-judge`. 3. a suite of the above two types of unit evaluators defined in `evaluation/evaluators.DEFINED_ANNOTATOR_SUITE_DICT`. Default to be suite `ahref` that we defined in our paper.
- `--config_dir`: the directory to contain configures for LLM-as-a-Judge evaluators.
- `--use_human_reference`: whether of not `annotator` needs to use the human reference. No need to specify if `annotator` specifies a evaluator suite. 
</details>


### Add a new evaluator
For this section, we give instructions on how to add a new evaluator `<new_evaluator>` that can be passed as the argument following `--annotator` for all commands. 

#### Add a non-LLM-based evaluator
1. Create a function for your evaluator in `href/evaluation/evaluators.py`.
2. Add the name `<new_evaluator>` to `href.evaluation.evaluators.DEFINED_ANNOTATORS`.

#### Add a LLM-based evaluator
To use LLM-as-a-Judge, we use a external package: a [modified version](https://github.com/tatsu-lab/alpaca_eval) of [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval). To create a new LLM-as-a-judge evaluator, we modify the configuration with the following steps:

##### 1. Create a new prompt template (Optional)
* Create a new prompt template under `href/llm-as-a-judge/prompt_templates`. 
* Note that besides the text, there are many placeholders of models' special tokens for different models to fit in, do not change their names. 
* Refer to the existing template to write new templates.

##### 2. Create a new model configuration
* Add the configuration for `<new_evaluator>` in `href/llm-as-a-judge/model_settings.json`. 
* The dictionary `template_kwargs` contains keys that corresponds to the placeholders in the prompt templates, please fill the values with the corresponding special token of your model.
* `fn_completions` and `completions_kwargs` are the configurations for the judge models. You can probably refer to the existing configurations for most of the desired setting. Please refer to [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) for more advanced settings.

##### 3. Create the configuration file 
To create the configuration file using the configurations from the previous two steps, run:
```bash
href create_config --model llama-70b --template_name basic_no_reference 
```
<details>
<summary> Required Arguments </summary>

- `--model_config_name`: the name of the model configuration used as the judge defined in `href/llm-as-a-judge/model_settings.json`.
- `--template_name`: the name of the template file in `href/llm-as-a-judge/prompt_templates` (without the suffix).
</details>

<details>
<summary> Optional Arguments </summary>

- `--config_dir`: the directory to save the resulting configuration.
- `--no_exmple`: if specified, remove the demonstration examples in the prompt.
- `--temperature`: the temperature for the judge model.
</details>