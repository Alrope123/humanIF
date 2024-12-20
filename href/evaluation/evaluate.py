import os
import json
import argparse
import logging
import random
from collections import defaultdict
import datasets
from alpaca_eval import evaluate as alpaca_farm_evaluate
from href.evaluation.evaluators import DEFINED_ANNOTATORS, ANNOTATOR_SUITE_DICT
import href.evaluation.evaluators as annotator_funcs

def evaluate(args):
    assert args.model_name is not None, "Model name should be specified."
    model_name = args.model_name

    # generate the model response if haven't
    if args.response_dir is None:
        from href.generation.generate import generate
        generate(args)
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
    if any([args.dataset.endswith(f".{suffix}") for suffix in ["csv, json, text"]]): # load from local file
        href_data = datasets.load_dataset(args.dataset.split(".")[-1], data_files=args.dataset)[args.split]
    else: # load from huggingface
        href_data = datasets.load_dataset(args.dataset)[args.split]
    baseline_responses = defaultdict(list)
    human_references = defaultdict(list)  # category -> list of example dicts
    for example in href_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        baseline_responses[category].append({
            "instruction": example['instruction'],
            "output": example['output'],
            "generator": "Meta-Llama-3.1-70B-Instruct",
            "dataset": f"href_{category}"
        })
        human_references[category].append({
            "instruction": example['instruction'],
            "output": example['reference'],
            "generator": "human",
            "dataset": f"href_{category}"
        })

    # specify the annotator for each category
    if args.annotator in ANNOTATOR_SUITE_DICT: # using different annotators for different category
        for category in args.nr_category:
            assert category in ANNOTATOR_SUITE_DICT[args.annotator], \
            f"Category {category} does not have an assigned annotator by {args.annotator}."
        category_to_annotator = ANNOTATOR_SUITE_DICT[args.annotator]
    else: # one single annotator for all categories
        category_to_annotator = {category: {
                                    'annotator': args.annotator, 
                                    'use_human_ref': args.use_human_reference} 
                                for category in args.nr_category}

    # running evaluation through AlpacaEval
    results = {"Average": {"wins": [], "ties": []}}
    for category in args.nr_category:
        annotator = category_to_annotator[category]['annotator']
        use_human_reference = category_to_annotator[category]['use_human_ref']

        category_model_responses = model_responses[category]
        category_baseline_responses = baseline_responses[category]
        if use_human_reference:
            category_human_references = human_references[category]
        else:
            category_human_references = None
        logging.info(f"Running evaluation on category: {category}")
        output_path = os.path.join(args.save_dir, model_name, category.lower().replace(" ", "_"))
        os.makedirs(output_path, exist_ok=True)

        
        if annotator == "perplexity":
            assert args.perplexity_path is not None, "Needs to specify a perplexity dir"
            evaluate_func = getattr(annotator_funcs, annotator)
            evaluate_func(category_baseline_responses, category_model_responses, category_human_references, category, args)
        elif annotator in DEFINED_ANNOTATORS: # non-llm annotators
            # run the according evaluation function
            evaluate_func = getattr(annotator_funcs, annotator)
            cur_annotations = evaluate_func(category_baseline_responses, category_model_responses, category_human_references, args)
            os.makedirs(os.path.join(output_path, annotator), exist_ok=True)
            json.dump(cur_annotations, open(os.path.join(output_path, annotator, "annotations.json"), 'w')) 
        else: # llm annotators
            cache_dir = os.path.join(args.cache_dir, model_name, category.lower().replace(" ", "_"))
            os.makedirs(cache_dir, exist_ok=True)
            alpaca_farm_evaluate(
                model_outputs=category_model_responses,
                reference_outputs=category_baseline_responses,
                human_outputs=category_human_references,
                annotators_config=annotator,
                output_path=output_path,
                is_return_instead_of_print=True,
                caching_path=os.path.join(cache_dir, f"{annotator}.json"),
                precomputed_leaderboard=None,
                is_cache_leaderboard=False,
                base_dir=args.config_dir,
                seed=args.seed,
                output_keys=("output_1", "output_2", "output_human") if use_human_reference else ("output_1", "output_2")
            )
            cur_annotations = json.load(open(os.path.join(output_path, annotator, "annotations.json"), 'r'))
        
        # we combined the results if coming from different basic annotators
        if annotator != args.annotator: 
            os.makedirs(os.path.join(output_path, args.annotator), exist_ok=True)
            json.dump(cur_annotations, open(os.path.join(output_path, args.annotator, "annotations.json"), 'w'))
        
        # record result 
        results[category] = {
            "wins": [cur_a['preference'] == 2.0 for cur_a in cur_annotations],
            "ties": [cur_a['preference'] == 0.0 for cur_a in cur_annotations]
        }
        results["Average"]["wins"].extend([cur_a['preference'] == 2.0 for cur_a in cur_annotations])
        results["Average"]["ties"].extend([cur_a['preference'] == 0.0 for cur_a in cur_annotations])
    
    for c, result in results.items():
        for t, annotations in result.items():
            result[t] = sum(annotations) / len(annotations)
        results[c] = result["wins"] + result["ties"] / 2 

    json.dump(results, open(os.path.join(args.save_dir, model_name, f"results_{args.annotator}.json"), 'w'))
    for category, score in results.items():
        logging.info(f"{category}: {score * 100 :.1f}")
    

def main():
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument(
        "--response_dir",
        type=str, 
        default=None,
        help="The directory that contains pre-generated model outputs. If specified, we will skip output generation and jump directly into evaluation."
    )
    parser.add_argument(
        "--perplexity_dir",
        type=str, 
        default=None,
        help="The directory that contains pre-generated model outputs. If specified, we will skip output generation and jump directly into evaluation."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="The model name that corresponds to the name of the yaml configuration file under `generation_config_dir` (exclude `.yaml`).",
    )
    parser.add_argument(
        "--generation_config_dir",
        type=str,
        default="href/generation/configs",
        help="The directory that contains the model generation configuration files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/href",
        help="The huggingface dataset name or the path to a local file to use for evaluation."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="The split to use in dataset."
    )
    parser.add_argument(
        "--nr_category",
        type=str,
        default=["Generation", "Open QA", "Brainstorm", "Rewrite", "Summarize",
                 "Classify", "Closed QA", "Extract"],
        nargs="+",
        help="Categories in the HREF to include."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results",
        help="Directory to save all results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="The directory to store downloaded datasets, models, and intermmediate annotation files.",
    )
    # evaluation arguments
    parser.add_argument(
        "--annotator",
        type=str,
        default="href",
        help="Name of the evaluation methods. It has to be one the three following: 1. a basic annotator defined in evaluation/evaluators.DEFINED_ANNOTATORS. 2. a configuration name for llm_as_a_judge that corresponds to a directory in llm_as_a_judge. 3. a suite of the above two types of unit evaluators defined in evaluation/evaluators.DEFINED_ANNOTATOR_SUITE_DICT`."
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="href/llm_as_a_judge/configs",
        help="The directory to contain configures for llm_as_a_judge evaluators",
    )
    parser.add_argument(
        "--use_human_reference",
        action="store_true",
        help="Whether of not annotator needs to use the human reference. No need to specify if annotator specifies a evaluator suite."
    )

    args = parser.parse_args()
    
    # set up
    random.seed(args.seed)
    os.environ['HF_HOME'] = args.cache_dir

    evaluate(args)

if __name__ == "__main__":
    main()