import os
import json
import argparse
import logging
import random
from collections import defaultdict
import datasets
from alpaca_eval import evaluate as alpaca_farm_evaluate
from href.evaluate.basic_annotators import DEFINED_ANNOTATORS, ANNOTATOR_GROUP_DICT
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
    assert args.annotator is not None and args.config_dir is not None, "Please specify the configuration of the annotator."

    # load model responses
    href_data = datasets.load_dataset(args.dataset)[args.split]
    data = defaultdict(list)
    for example in href_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        data[category].append(example)

    # specify the annotator for each category
    if args.annotator in ANNOTATOR_GROUP_DICT: # using different annotators for different category
        for category in args.nr_category:
            assert category in ANNOTATOR_GROUP_DICT[args.annotator], \
            f"Category {category} does not have an assigned annotator by {args.annotator}."
        category_to_annotator = ANNOTATOR_GROUP_DICT[args.annotator]
    else: # one single annotator for all categories
        category_to_annotator = {category: {
                                    'annotator': args.annotator, 
                                    'use_human_ref': args.use_human_reference} 
                                for category in args.nr_category}

    # running evaluation through AlpacaEval
    for category in args.nr_category:
        annotator = category_to_annotator[category]['annotator']
        use_human_reference = category_to_annotator[category]['use_human_ref']

        category_responses_a = [{
            "instruction": dp['instruction'],
            "output": dp['output_a'],
            "generator": dp['generator_a'],
            "dataset": f"href_{category}"
        } for dp in data[category]]
        category_responses_b = [{
            "instruction": dp['instruction'],
            "output": dp['output_b'],
            "generator": dp['generator_b'],
            "dataset": f"href_{category}"
        } for dp in data[category]]
        if use_human_reference:
            category_human_references = [{
                "instruction": dp['instruction'],
                "output": dp['reference'],
                "generator": "human",
                "dataset": f"href_{category}"
            } for dp in data[category]]
        else:
            category_human_references = None
        logging.info(f"Running evaluation on category: {category}")
        output_path = os.path.join(args.save_dir, "human_agreement_analysis", category.lower().replace(" ", "_"))
        os.makedirs(output_path, exist_ok=True)        

        if annotator in DEFINED_ANNOTATORS: # non-llm annotators
            # run the according evaluation function
            evaluate_func = getattr(annotator, annotator)
            cur_annotations = evaluate_func(category_responses_a, category_responses_b, category_human_references, args)
            os.makedirs(os.path.join(output_path, annotator), exist_ok=True)
            json.dump(cur_annotations, open(os.path.join(output_path, annotator, "annotations.json"), 'w')) 
        else: # llm annotators
            cache_dir = os.path.join(args.cache_dir, "human_agreement_analysis", category.lower().replace(" ", "_"))
            os.makedirs(cache_dir, exist_ok=True)
            alpaca_farm_evaluate(
                model_outputs=category_responses_a,
                reference_outputs=category_responses_b,
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


        # calculate agreement with human
        cur_annotations = json.load(open(os.path.join(output_path, args.annotator, "annotations.json"), 'r'))
        prompt_to_annotation = {}
        for a in cur_annotations:
            prompt_to_annotation[a["instruction"], a["generator_2"], a["generator_1"]] = int(a["preference"])
        rates_inner = []
        rates_outer = []
        for dp in data[category]:
            rates_inner.append(leave_one_out_agreement_inner(dp['annotations']))
            rates_outer.append(leave_one_out_agreement_outer(dp['annotations'], prompt_to_annotation[dp['instruction'], dp['generator_a'], dp['generator_b']]))
        agreement_rate_inner = sum(rates_inner) / len(rates_inner)
        agreement_rate_outer = sum(rates_outer) / len(rates_outer)
        logging.info(f"Category: {category}")
        logging.info(f"Inner human agreement rate: {agreement_rate_inner * 100 : .1f}%")
        logging.info(f"Human agreement rate using {args.annotator}: {agreement_rate_outer * 100 : .1f}%")
        
    
def main():
    parser = argparse.ArgumentParser()
    # evaluation arguments
    parser.add_argument(
        "--annotator",
        type=str,
        default="basic_no_reference_gpt4",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="href/LLM-as-a-Judge/configs",
        help="If specified, we will use the dir as the root directory for annotator configuration.",
    )
    parser.add_argument(
        "--use_human_reference",
        action="store_true",
        help="If given, we will embed human response into the prompt."
    )
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
        default="train",
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
        default="results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="The directory to store downloaded datasets and models.",
    )

    args = parser.parse_args()
    
    # set up
    random.seed(args.seed)
    os.environ['HF_HOME'] = args.cache_dir

    evaluate(args)

if __name__ == "__main__":
    main()