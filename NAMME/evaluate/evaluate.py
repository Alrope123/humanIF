import os
import json
import argparse
import logging
import random
from collections import defaultdict
import datasets
from alpaca_eval import evaluate as alpaca_farm_evaluate
from NAMME.evaluate.basic_annotators import DEFINED_ANNOTATORS, ANNOTATOR_DICT
from generate.generate import generate

def evaluate(args):
    assert args.annotator is not None and args.config_dir is not None, "Please specify the configuration of the annotator."
    assert (args.model_name_or_path is not None) or (args.openai_engine is not None), "Either model_name_or_path or openai_engine should be specified."
    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None \
        else args.openai_engine + f"-t={args.temperature}" 

    # generate the model response if haven't
    if args.response_dir is None:
        generate(args)
        model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None \
            else args.openai_engine + f"-t={args.temperature}"
        logging.info(f"Generating responses from {model_name}")
        response_dir = os.path.join(args.save_dir, model_name)
    else:
        logging.info(f"Using responses from {args.response_dir}")
        response_dir = args.response_dir

    # load model generated response
    model_responses = defaultdict(list)
    for category in args.nr_category:
        with open(os.path.join(response_dir, category.lower().replace(" ", "_"), "responses.jsonl"), "r") as fin:
            model_responses[category].extend([json.loads(line) for line in fin])

    # load baseline model response and/or human response
    NAMME_data = datasets.load_dataset(args.dataset)[args.split]
    baseline_responses = defaultdict(list)
    if args.use_human_reference:
        human_references = defaultdict(list)  # category -> list of example dicts
    for example in NAMME_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        baseline_responses[category].append({
            "instruction": example['instruction'],
            "output": example['output'],
            "generator": "Meta-Llama-3.1-70B-Instruct",
            "dataset": f"NAMME_{category}"
        })
        if args.use_human_reference:
            human_references[category].append({
                "instruction": example['instruction'],
                "output": example['reference'],
                "generator": "human",
                "dataset": f"NAMME_{category}"
            })

    # specify the annotator for each category
    if args.annotator in ANNOTATOR_DICT: # using different annotators for different category
        for category in args.nr_category:
            assert category in ANNOTATOR_DICT[args.annotator], \
            f"Category {category} does not have an assigned annotator by {args.annotator}."
        category_to_annotator = ANNOTATOR_DICT[args.annotator]
    else:
        category_to_annotator = {category: args.annotator for category in args.nr_category}

    # running evaluation through AlpacaEval
    results = {}
    for category in args.nr_category:
        category_model_responses = model_responses[category]
        category_baseline_responses = baseline_responses[category]
        if args.use_human_reference:
            category_human_references = human_references[category]
        else:
            category_human_references = None
        logging.info(f"Running evaluation on category: {category}")
        output_path = os.path.join(args.save_dir, model_name, category.lower().replace(" ", "_"))
        os.makedirs(output_path, exist_ok=True)
        annotator = category_to_annotator[category]

        if annotator in DEFINED_ANNOTATORS: # non-llm annotators
            # run the according evaluation function
            evaluate_func = getattr(annotator, annotator)
            cur_annotations = evaluate_func(category_model_responses, category_baseline_responses, category_human_references, args)
            os.makedirs(os.path.join(output_path, annotator), exist_ok=True)
            json.dump(cur_annotations, open(os.path.join(output_path, annotator, "annotations.json"), 'w')) 
        else: # llm annotators
            alpaca_farm_evaluate(
                model_outputs=category_model_responses,
                reference_outputs=category_baseline_responses,
                human_outputs=category_human_references,
                annotators_config=annotator,
                output_path=output_path,
                is_return_instead_of_print=True,
                precomputed_leaderboard=None,
                is_cache_leaderboard=False,
                base_dir=args.config_dir,
                seed=args.seed,
                output_keys=("output_1", "output_2", "output_human") if args.use_human_reference else ("output_1", "output_2")
            )
            cur_annotations = json.load(open(os.path.join(output_path, annotator, "annotations.json"), 'r'))
        
        # we combined the results if coming from different basic annotators
        if annotator != args.annotator: 
            os.makedirs(os.path.join(output_path, args.annotator), exist_ok=True)
            json.dump(cur_annotations, open(os.path.join(output_path, args.annotator, "annotations.json"), 'w'))
        
        # summarize result 
        positive_annotations = [cur_a['preference'] in [2.0, 0.0] for cur_a in cur_annotations]
        score = sum(positive_annotations) / len(positive_annotations)
        results[category] = score
    
    results["Average"] = sum(results.values()) / len(results)
    json.dump(results, open(os.path.join(model_name, f"results_{args.annotator}.json"), 'r'))
    for category, score in results.items():
        logging.info(f"{category}: {score * 100 :.1f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # generation arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/no_robots",
        help="Path to the reference outputs. If none is provided, will use human-written references."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="The split of the dataset to use."
    )
    parser.add_argument(
        "--nr_category",
        type=str,
        default=["Generation", "Open QA", "Brainstorm", "Chat", "Rewrite", "Summarize",
                 "Coding", "Classify", "Closed QA", "Extract", "Fact Checking or Attributed QA", "Multi-Document Synthesis", "Reasoning Over Numerical Data"],
        nargs="+",
        help="Categories in the No Robots dataset to include. If not specified, all categories will be used"
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results/alpaca_farm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default="gpt-3.5-turbo",
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="generate.templates.create_prompt_with_huggingface_tokenizer_template", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature we use for model generation.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="The directory to store downloaded datasets and models.",
    )
    # evaluation arguments
    parser.add_argument(
        "--response_dir",
        type=str, 
        default="results/alpaca_farm",
        help="If specified, we will load the model responses from the directory"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="If specified, we will use the dir as the root directory for annotator configuration.",
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default="basic_no_reference_gpt4",
    )
    parser.add_argument(
        "--use_human_reference",
        action="store_true",
        help="If given, we will embed human response into the prompt."
    )
    args = parser.parse_args()
    
    # set up
    random.seed(args.seed)
    os.environ['HF_HOME'] = args.cache_dir

    evaluate(args)
