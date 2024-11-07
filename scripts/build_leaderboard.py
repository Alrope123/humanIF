import os
import json
import argparse
import random
import csv
import numpy as np
from collections import defaultdict
from scipy import stats

def get_pvalue_from_paired_t_test(annotation1, annotation2):
    ttest_result = stats.ttest_rel(annotation1, annotation2)
    return ttest_result.pvalue


def decide_is_distinguishable(annotation1, annotation2):
    return get_pvalue_from_paired_t_test(annotation1, annotation2) <= 0.05


def main(args):
    random.seed(42)
    nr_category = args.nr_category
    result_dir = args.result_dir
    models = args.models
    annotator = args.annotator
    save_dir = args.save_dir

    # initialize csv file
    os.makedirs(save_dir, exist_ok=True)
    leaderboard_csv = csv.writer(open(os.path.join(save_dir, "leaderboard.csv"), "w+"))
    titles = ["Model"] + nr_category + ["Average"] + [f"{cat} Rank" for cat in nr_category] + ["Average Rank"]
    leaderboard_csv.writerow(titles)

    # read annotations
    positive_rates = defaultdict(list)
    annotations = defaultdict(list)
    for model in models:
        for category in nr_category:
            category_name = category.lower().replace(' ', '_')
            cur_annotations = json.load(open(os.path.join(result_dir, model, category_name, annotator, "annotations.json"), 'r'))
            cur_annotations = [cur_a['preference'] for cur_a in cur_annotations]
            positive_annotations = [cur_a in [2.0, 0.0] for cur_a in cur_annotations]
            positive_rates[model].append(sum(positive_annotations) / len(positive_annotations))
            cur_annotations = [cur_a if cur_a is not None else -1 for cur_a in cur_annotations]
            annotations[model].append(cur_annotations)

        average_positive_rate = np.mean(positive_rates[model])
        model_annotations = [a for cat_annotations in annotations[model] for a in cat_annotations]
        positive_rates[model].append(average_positive_rate)
        annotations[model].append(model_annotations)

    # calculate ranking
    rankings = defaultdict(list)
    for cat_index, cat in enumerate((nr_category + ["Average"])):
        # sort data by average
        sorted_positive_rates = dict(sorted(positive_rates.items(), key=lambda x: x[1][cat_index], reverse=True))
        sorted_annotations = [[m] + annotations[m] for m in sorted_positive_rates]

        # decide distinguishibility ranking
        i = 0
        while i < len(sorted_annotations):
            model_name = sorted_annotations[i][0]
            cur_annotation_list = sorted_annotations[i][cat_index+1]
            # decide if the following model is distinguishable from the current
            rankings[model_name].append(i+1)
            newly_added = 0
            for j in range(i + 1, len(sorted_annotations), 1):
                next_model_name = sorted_annotations[j][0]
                next_annotation_list = sorted_annotations[j][cat_index+1]
                if not decide_is_distinguishable(cur_annotation_list, next_annotation_list):
                    rankings[next_model_name].append(i+1)
                    newly_added += 1
                else:
                    break
            i += 1 + newly_added
                        
    # add rankings and sort by average finally
    sorted_positive_rates = dict(sorted(positive_rates.items(), key=lambda x: x[1][-1], reverse=True))
    sorted_results = [[m] + rates + rankings[m] for m, rates in sorted_positive_rates.items()]

    # write to the files
    for row in sorted_results:
        leaderboard_csv.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        nargs="+",
        help="Name of the models to build the leaderboard."
    )

    parser.add_argument(
        "--annotator",
        type=str,
        default='NAMME',
        help="Name of the evaluator that generated the annotation results."
    )

    parser.add_argument(
        "--nr_category",
        type=str,
        nargs="+",
        default=["Generation", "Open QA", "Brainstorm", "Rewrite", "Summarize",
                 "Classify", "Closed QA", "Extract"],
        help="Categories in the HREF to include."
    )
    
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Path to the dir that contains the annotation files."
    )

    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results",
        help="Directory to save all results."
    )

    args = parser.parse_args()
    main(args)