import os
import json
import argparse
import random
import csv
import numpy as np
from collections import defaultdict
from scipy import stats
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def bootstrap(data, num_resamples=5000, statistic=np.mean, seed=None):
    """
    Perform bootstrap resampling on a list of scores.

    Parameters:
    - data (list or array): The data to bootstrap (e.g., list of scores).
    - num_resamples (int): The number of bootstrap samples to generate.
    - statistic (function): The statistic to compute on each resample (e.g., np.mean, np.median).
    - seed (int or None): Seed for the random number generator (optional).

    Returns:
    - bootstrapped_stats (array): An array of the computed statistic for each bootstrap sample.
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = np.array(data)
    bootstrapped_stats = []

    for _ in range(num_resamples):
        # Resample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the statistic on the resample
        bootstrapped_stat = statistic(sample)
        bootstrapped_stats.append(bootstrapped_stat)

    lower_bound = np.percentile(bootstrapped_stats, 2.5)
    upper_bound = np.percentile(bootstrapped_stats, 97.5)
    return np.array(bootstrapped_stats), lower_bound, upper_bound


# compute pvalue of the t-test between 
def get_pvalue_from_paired_t_test(annotation1, annotation2):
    ttest_result = stats.ttest_rel(annotation1, annotation2)
    return ttest_result.pvalue

# test if significant
def decide_is_distinguishable(annotation1, annotation2):
    return get_pvalue_from_paired_t_test(annotation1, annotation2) <= 0.05


def decide_is_distinguishable(ci1, ci2):
    return ci1[3] > ci2[2] or ci1[2] < ci2[3]


# create a p-values v.s. data size plot 
def generate_ttest_plot(annotations, save_path, interval=100, percentile=90):
    # randomly shuffle the annotations
    data_size = len(next(iter(annotations.values())))
    indices = list(range(data_size))
    random.shuffle(indices)
    annotations = {key: [value[i] for i in indices] for key, value in annotations.items()}
    
    # get p-values across different size of data to use
    p_value_indicators = [1]
    sizes = [0]
    for size in range(interval, data_size, interval):
        p_values = []
         # calculate all pairwise p-values
        for m1, m2 in combinations(annotations.keys(), 2):
            p_values.append(get_pvalue_from_paired_t_test(annotations[m1][:size], annotations[m2][:size]))
        p_values = sorted(p_values)
        p_value_indicators.append(np.percentile(p_values, percentile))
        sizes.append(size)

    # plotting    
    fontdict={'fontsize': 15}
    
    data = pd.DataFrame({
        'x': sizes,
        'y': p_value_indicators
    })
    sns.set_theme(style="white")  # Optional: set the style of the plot
    sns.lineplot(x='x', y='y', data=data, color="#66c2a5", label=f"{percentile}% Percentile")
    
    plt.axhline(y=0.05, color='grey', linestyle='--', linewidth=1.5, label='0.05')
    
    plt.xlabel("Size", fontdict=fontdict)
    plt.ylabel("P-Value", fontdict=fontdict)
    plt.legend(prop={'size': 15})

    plt.savefig(save_path)


def calculate_win_rate(annotations):
    win_annotations = [cur_a in [2.0] for cur_a in annotations]
    tie_annotations = [cur_a in [0.0] for cur_a in annotations]
    return sum(win_annotations) / len(win_annotations) + sum(tie_annotations) / len(tie_annotations) / 2


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
    # confidence_interval=defaultdict(list)
    for model in tqdm(models):
        for category in nr_category:
            category_name = category.lower().replace(' ', '_')
            cur_annotations = json.load(open(os.path.join(result_dir, model, category_name, annotator, "annotations.json"), 'r'))
            cur_annotations = [cur_a['preference'] for cur_a in cur_annotations]
            positive_rate = calculate_win_rate(cur_annotations)
            positive_rates[model].append(positive_rate)
            # _, lower, upper = bootstrap(cur_annotations, statistic=calculate_win_rate)
            # confidence_interval[model].append((upper-positive_rate, lower-positive_rate, upper, lower))
            cur_annotations = [cur_a if cur_a is not None else -1 for cur_a in cur_annotations]
            annotations[model].append(cur_annotations)

        model_annotations = [a for cat_annotations in annotations[model] for a in cat_annotations]
        average_positive_rate = calculate_win_rate(model_annotations)
        positive_rates[model].append(average_positive_rate)
        # _, lower, upper = bootstrap(model_annotations, statistic=calculate_win_rate)
        # confidence_interval[model].append((upper-average_positive_rate, lower-average_positive_rate, upper, lower))
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

    # rankings = defaultdict(list)
    # for cat_index, cat in enumerate((nr_category + ["Average"])):
    #     # sort data by average
    #     sorted_positive_rates = dict(sorted(positive_rates.items(), key=lambda x: x[1][cat_index], reverse=True))
    #     sorted_ci = [[m] + confidence_interval[m] for m in sorted_positive_rates]

    #     # decide distinguishibility ranking
    #     i = 0
    #     while i < len(sorted_ci):
    #         model_name = sorted_ci[i][0]
    #         cur_confidence_interval = sorted_ci[i][cat_index+1]
    #         # decide if the following model is distinguishable from the current
    #         rankings[model_name].append(i+1)
    #         newly_added = 0
    #         for j in range(i + 1, len(sorted_ci), 1):
    #             next_model_name = sorted_ci[j][0]
    #             next_confidence_interval = sorted_ci[j][cat_index+1]
    #             if not decide_is_distinguishable(cur_confidence_interval, next_confidence_interval):
    #                 rankings[next_model_name].append(i+1)
    #                 newly_added += 1
    #             else:
    #                 break
    #         i += 1 + newly_added

    # add rankings and sort by average finally
    sorted_positive_rates = dict(sorted(positive_rates.items(), key=lambda x: x[1][-1], reverse=True))
    sorted_results = [[m] + rates + rankings[m] for m, rates in sorted_positive_rates.items()]

    print(json.dumps(sorted_results, indent=4))

    # write to the files
    for row in sorted_results:
        leaderboard_csv.writerow(row)

    # # output a p-value plot
    # generate_ttest_plot({m: annotation_list[-1] for m, annotation_list in annotations.items()}, 
    #                     os.path.join(args.save_dir, "p-value_plot.png"))
    
    # # Latex code
    # int_to_letter = {
    #     '0': 'a', '1': 'b', '2': 'c', '3': 'd', '4': 'e',
    #     '5': 'f', '6': 'g', '7': 'h', '8': 'i', '9': 'j'
    # }
    # import re
    # SUFFIX = {
    #     "Brainstorm": "b",         
    #     "Generation": "g",         
    #     "Rewrite": "r",                  
    #     "Open QA": "oq",                  
    #     "Classify": "c",
    #     "Summarize": "s",
    #     "Extract": "e",
    #     "Closed QA": "cq",
    #     'Fact Checking or Attributed QA': "f",
    #     'Multi-Document Synthesis': "m",
    #     'Reasoning Over Numerical Data': "ro",
    #     "Micro Average": "o",
    #     "Ranking": "rk" 
    # }

    # # split = "dev"
    # # split = "test"

    # all_commands = []
    # table_row = []
    # for model in sorted_positive_rates:
    #     # Generate Overleaf
    #     overall_outputs = {} 
    #     for i, category in enumerate(nr_category):
    #         overall_outputs[category] = positive_rates[model][i]

    #     overall_outputs["Micro Average"] = positive_rates[model][-1]
    #     overall_outputs["Ranking"] = rankings[model][-1]
    #     model_prefix = model.split("-t=")[0].replace('-','').replace("_", '').lower()
    #     model_prefix = ''.join(int_to_letter[char] if char in int_to_letter else char for char in model_prefix)
    #     model_prefix = re.sub(r'[\d\.]', '', model_prefix)
        
    #     command_template = "\\newcommand{\cmd}{num}"
    #     categorical_commands = []
    #     for category, num in overall_outputs.items():
    #         if type(num) != int:
    #             num = str(round(num * 100, 1))
    #         else:
    #             num = str(num)
    #             # continue
    #         categorical_command = model_prefix + SUFFIX[category] + ("d" if split == "dev" else "") 
    #         all_commands.append(command_template.replace('cmd', categorical_command).replace('num', num))
    #         categorical_commands.append(categorical_command)
        
    #     row_string = model.split("-t=")[0] + f"({overall_outputs['Ranking']})"
    #     for i, cmd in enumerate(categorical_commands):
    #         if i == len(categorical_commands) - 1:
    #             break
    #         row_string += f" & \{cmd}"
    #     row_string += " \\\\"
    #     table_row.append(row_string)


    # for cmd in all_commands:
    #     print(cmd)

    # for row in table_row:
    #     print(row)


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
        default='ahref',
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