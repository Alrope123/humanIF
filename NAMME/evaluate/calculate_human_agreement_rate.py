import os
import json
import argparse
import logging
import random
from collections import defaultdict
import datasets
from alpaca_eval import evaluate as alpaca_farm_evaluate
from collections import Counter


def calculate_mode(annotations):
    """Calculate the mode of the annotations. If multiple modes exist, choose one at random."""
    count = Counter(annotations)
    max_freq = max(count.values())
    modes = [key for key, freq in count.items() if freq == max_freq]
    
    # Randomly choose one if there are multiple modes
    return random.choice(modes)

def leave_one_out_agreement_inner(annotations):
    """Compute the leave-one-out agreement for a list of annotations."""
    num_annotations = len(annotations)
    correct_predictions = 0
    
    for i in range(num_annotations):
        # Leave one out
        remaining_annotations = annotations[:i] + annotations[i+1:]
        
        # Calculate the mode of the remaining annotations
        mode = calculate_mode(remaining_annotations)
        
        # Check if the prediction matches the mode
        if mode == annotations[i]:
            correct_predictions += 1
    
    # Return the accuracy (agreement rate) for this set of annotations
    return correct_predictions / num_annotations


def leave_one_out_agreement_outer(annotations, my_p):
    """Compute the leave-one-out agreement for a list of annotations."""
    num_annotations = len(annotations)
    correct_predictions = 0
    
    for i in range(num_annotations):
        # Leave one out
        remaining_annotations = annotations[:i] + annotations[i+1:]
        
        # Calculate the mode of the remaining annotations
        mode = calculate_mode(remaining_annotations)
        
        # Check if the prediction matches the mode
        if mode == my_p:
            correct_predictions += 1
    
    # Return the accuracy (agreement rate) for this set of annotations
    return correct_predictions / num_annotations

def LOO_agreement(human_prefs, my_p):
    accs = []
    for i, _ in enumerate(human_prefs):
        for j, pref2 in enumerate(human_prefs):
            if i != j:
                accs.append(my_p == pref2)
    return sum(accs) / len(accs)


def evaluate(args):
    assert args.config_name is not None and args.config_dir is not None, "Please specify the configuration of the annotator."
    assert (args.model_name_or_path is not None) or (args.openai_engine is not None), "Either model_name_or_path or openai_engine should be specified."
    model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None else args.openai_engine

    # load model responses
    NAMME_data = datasets.load_dataset(args.dataset)[args.split]
    data = defaultdict(list)
    for example in NAMME_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        data[category].append(example)

    # running evaluation through AlpacaEval
    results = {}
    for category in args.nr_category:
        category_responses_a = [dp['output_1'] for dp in data[category]]
        category_responses_b = [dp['output_2'] for dp in data[category]]
        if args.use_human_reference:
            category_human_references = [dp['human_reference'] for dp in data[category]]
        else:
            category_human_references = None
        logging.info(f"Running evaluation on category: {category}")
        model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None \
            else args.openai_engine + f"-t={args.temperature}"
        output_path = os.path.join(args.save_dir, model_name, category.lower().replace(" ", "_"))
        os.makedirs(output_path, exist_ok=True)
        alpaca_farm_evaluate(
            model_outputs=category_responses_a,
            reference_outputs=category_responses_b,
            human_outputs=category_human_references,
            annotators_config=args.config_name,
            output_path=output_path,
            is_return_instead_of_print=True,
            precomputed_leaderboard=None,
            is_cache_leaderboard=False,
            base_dir=args.config_dir,
            seed=args.seed,
            output_keys=("output_1", "output_2", "output_human") if args.use_human_reference else ("output_1", "output_2")
        )
    
    # calculate agreement with human
    cur_annotations = json.load(open(os.path.join(output_path, args.config_name, "annotations.json"), 'r'))
    prompt_to_annotation = {}
    for a in cur_annotations:
        prompt_to_annotation[a["instruction"], a["generator_1"], a["generator_2"]] = int(a)
    rates_inner = []
    rates_outer = []
    for dp in data:
        rates_inner.append(leave_one_out_agreement_inner(dp['annotations']))
        rates_outer.append(leave_one_out_agreement_outer(dp['annotations'], prompt_to_annotation[dp['instruction'], dp['model_1'], dp['model_2']]))
    agreement_rate_inner = sum(rates_inner) / len(rates_inner)
    agreement_rate_outer = sum(rates_outer) / len(rates_outer)
    logging.info(f"Inner human agreement rate: {agreement_rate_inner * 100 : .1f}")
    logging.info(f"Human agreement rate using {args.config_name}: {agreement_rate_outer * 100 : .1f}")
        
    
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
        "--config_name",
        type=str,
        default="alpaca_eval_gpt4",
    )
    parser.add_argument(
        "--use_human_reference",
        action="store_true",
        help="If given, we will embed human response into the prompt."
    )
    # human agreement rate arguments

    args = parser.parse_args()
    
    # set up
    random.seed(args.seed)
    os.environ['HF_HOME'] = args.cache_dir

    evaluate(args)
