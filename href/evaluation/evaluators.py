from collections import defaultdict

ANNOTATOR_SUITE_DICT = {
    "href": {
        "Brainstorm": {
            "annotator": "llama3.1-70b_basic_no_reference",
            "use_human_ref": False,
        },
        "Open QA": {
            "annotator": "bertscore",
            "use_human_ref": True,
        },
        "Closed QA": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Extract": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Generation": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Rewrite": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Summarize": {
            "annotator": "llama3.1-70b_basic_no_reference",
            "use_human_ref": False,
        },
        "Classify": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Fact Checking or Attributed QA": {
            "annotator": "bertscore",
            "use_human_ref": True,
        },
        "Multi-Document Synthesis": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Reasoning Over Numerical Data": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
    },
    "href_7b": {
        "Brainstorm": {
            "annotator": "llama3.1_basic_no_reference",
            "use_human_ref": False,
        },
        "Open QA": {
            "annotator": "bertscore",
            "use_human_ref": True,
        },
        "Closed QA": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Extract": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
        "Generation": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
        "Rewrite": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
        "Summarize": {
            "annotator": "llama3.1_basic_no_reference",
            "use_human_ref": False,
        },
        "Classify": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
        "Fact Checking or Attributed QA": {
            "annotator": "bertscore",
            "use_human_ref": True,
        },
        "Multi-Document Synthesis": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Reasoning Over Numerical Data": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
    },
}

ANNOTATION_REVERSE_MAP = {
    1: 2,
    2: 1,
    0: 0
}

DEFINED_ANNOTATORS = ["short", "long", "random_no_tie", "bertscore", "rouge"]

def bertscore(responses_1, responses_2, human_references, args):
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en")
    # calculate score
    _, _, f1s_1 = scorer.score([res['output'] for res in responses_1], [res['output'] for res in human_references])
    _, _, f1s_2 = scorer.score([res['output'] for res in responses_2], [res['output'] for res in human_references])
    annotations = []
    for res1, res2, human_ref, f1_1, f1_2 in zip(responses_1, responses_2, human_references, f1s_1, f1s_2):
        if f1_1 > f1_2:
            score = 1.0
        elif f1_1 < f1_2:
            score = 2.0
        else:
            score = 0.0

        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "output_human": human_ref["output"],
            "annotator": "bert",
            "preference": score
        })
    return annotations


def rouge(model_responses, baseline_responses, human_references, args):
    from rouge import Rouge
    rouge = Rouge()

    annotations = []
    for res1, res2, human_ref in zip(model_responses, baseline_responses, human_references):
        # get rouge scores 
        if len(res1['output'].strip()) == 0:
            eval_1_rouge = [{"rouge-1": {'f': 0}}]
        else:
            eval_1_rouge = rouge.get_scores(res1['output'], human_ref['output'])
        if len(res2['output'].strip()) == 0:
            eval_2_rouge = [{"rouge-1": {'f': 0}}]
        else:
            eval_2_rouge = rouge.get_scores(res2['output'], human_ref['output'])

        eval_1_score = eval_1_rouge[0]["rouge-1"]["f"]
        eval_2_score = eval_2_rouge[0]["rouge-1"]["f"]
        if eval_1_score > eval_2_score:
            score = 1.0
        elif eval_1_score < eval_2_score:
            score = 2.0
        else:
            score = 0.0
        
        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "output_human": human_ref["output"],
            "annotator": "rouge-1",
            "preference": score
        })
    return annotations


def long(model_responses, baseline_responses, human_references, args):
    annotations = []
    for res1, res2 in zip(model_responses, baseline_responses):
        if len(res1["output"]) > len(res2["output"]):
            score = 1.0
        elif len(res1["output"]) < len(res2["output"]):
            score = 2.0
        else:
            score = 0.0

        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "annotator": "long",
            "preference": score
        })
    return annotations


def short(model_responses, baseline_responses, human_references, args):
    annotations = []
    for res1, res2 in zip(model_responses, baseline_responses):
        if len(res1["output"]) < len(res2["output"]):
            score = 1.0
        elif len(res1["output"]) > len(res2["output"]):
            score = 2.0
        else:
            score = 0.0
        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "annotator": "short",
            "preference": score
        })
    return annotations


def random_no_tie(model_responses, baseline_responses, human_references, args):
    import random
    annotations = []
    for res1, res2 in zip(model_responses, baseline_responses):
        score = 1.0 if bool(random.getrandbits(1)) else 2.0
        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "annotator": "short",
            "preference": score
        })
    return annotations


def perplexity(model_responses, baseline_responses, human_references, category, args):
    import os
    import json

    generator_model = model_responses[0]["generator"]
    generator_baseline = baseline_responses[0]["generator"]
    # load model generated perplexities
    model_perplexities = defaultdict(list)
    with open(os.path.join(args.perplexity_dir, generator_model, category.lower().replace(" ", "_"), "loss.jsonl"), "r") as fin:
        model_perplexities[category].extend([json.loads(line) for line in fin])
    baseline_perplexities = defaultdict(list)
    with open(os.path.join(args.perplexity_dir, generator_baseline, category.lower().replace(" ", "_"), "loss.jsonl"), "r") as fin:
        baseline_perplexities[category].extend([json.loads(line) for line in fin])
    
    annotations = []
    for res1, ppl1, res2, ppl2, res_human in zip(model_responses, model_perplexities, baseline_responses, baseline_perplexities, human_references):
        assert ppl1['reference'] == ppl2['reference'] and ppl1['reference'] == res_human['output'], "Plexities doesn't match"
        if ppl1["output"] < ppl2["output"]:
            score = 1.0
        if ppl1["output"] > ppl2["output"]:
            score = 2.0
        else:
            score = 0.0

        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "annotator": "perplexity",
            "preference": score
        })

    return annotations