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
    distinguishability_csv = csv.writer(open(os.path.join(save_dir, "distinguishability.csv"), "w+"))
    titles = ["Model"] + nr_category + ["Average"]
    leaderboard_csv.writerow(titles)
    distinguishability_csv.writerow(titles)

    # read annotations
    positive_rates = defaultdict(list)
    annotations = defaultdict(list)
    for model in models:
        for category in nr_category:
            category_name = category.lower().replace(' ', '_')
            cur_annotations = json.load(open(os.path.join(result_dir, model, category_name, annotator, "annotations.json"), 'r'))
            cur_annotations = [cur_a['preference'] for cur_a in cur_annotations]
            positive_annotations = [cur_a['preference'] in [2.0, 0.0] for cur_a in cur_annotations]
            positive_rates[model].append(sum(positive_annotations) / len(positive_annotations))
            cur_annotations = [cur_a if cur_a is not None else -1 for cur_a in cur_annotations]

        average_positive_rate = np.mean(positive_rates[model].values())
        model_annotations = [a for cat_annotations in annotations[model].values() for a in cat_annotations]
        positive_rates[model].append(average_positive_rate)
        annotations[model].append(model_annotations)

    # sort data by average
    sorted_positive_rates = dict(sorted(positive_rates.items(), key=lambda x: x[1][-1], reverse=True))
    sorted_annotations = [[m] + annotations[m] for m in sorted_positive_rates]

    # decide distinguishibility
    sorted_distinguishabilities = []
    for i, annotations_lists in enumerate(sorted_annotations):
        cur_distinguishabilities = []
        for j, annotation_list in enumerate(annotations_lists):
            if j == 0: # simply add the model identifier
                cur_distinguishabilities.append(annotation_list)
                continue
            # decide if the model is distinguishable from other models
            indistinguishable_models = []
            for k in range(len(sorted_annotations)):
                if i != k:
                    other_annotation_list = sorted_annotations[k][j]
                    if not decide_is_distinguishable(other_annotation_list, annotation_list):
                        indistinguishable_models.append(sorted_annotations[k][0])
            cur_distinguishabilities.append(",".join(indistinguishable_models))
        sorted_distinguishabilities.append(cur_distinguishabilities)

    # write to the files
    for m, rates in sorted_positive_rates.items():
        leaderboard_csv.writerow([m] + rates)
    for distinguishability_list in sorted_distinguishabilities:
        distinguishability_csv.writerow(distinguishability_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        nargs="+",
        help="Path to the dir that contains the prediction file."
    )

    parser.add_argument(
        "--nr_category",
        type=str,
        nargs="+",
        help="Categories in the No Robots dataset to include. If not specified, all categories will be used"
    )
    
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Path to the dir that contains the prediction file."
    )

    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results"
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        required=True,
        help="Path to the dir that contains the prediction file."
    )



    parser.add_argument(
        "--annotator",
        type=str,
        default='llama3.1-70b_basic_w_reference',
        help="Path to the dir that contains the prediction file."
    )


    args = parser.parse_args()
    main(args)