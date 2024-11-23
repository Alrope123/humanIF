import os
import json
import argparse
import logging
import random
from collections import defaultdict
import torch
import vllm
import datasets
import yaml
import math
from openai import OpenAI
import time

from href.generation.utils import generate_perplexities, create_prompt_with_huggingface_tokenizer_template, load_hf_lm, load_hf_tokenizer

def find_shared_segment(str1, str2):
    reversed_str1 = str1[::-1]
    reversed_str2 = str2[::-1]
    
    # Find the common segment
    shared_segment = []
    for char1, char2 in zip(reversed_str1, reversed_str2):
        if char1 == char2:
            shared_segment.append(char1)
        else:
            break

    # Reverse the result to get the correct shared segment
    return ''.join(shared_segment[::-1])

def find_sublist_indices(main_list, sublist):
    sublist_len = len(sublist)
    main_list_len = len(main_list)
    # Iterate through the main list to find the starting position of the sublist
    for i in range(main_list_len - sublist_len + 1):
        if main_list[i:i + sublist_len] == sublist:
            return i, i + sublist_len
    # If the sublist is not found, return None
    assert False, [main_list, sublist]


def generate_loss(args):
    assert args.model_name is not None, "Model name should be specified."
    model_name = args.model_name
    config_path = os.path.join(args.generation_config_dir, f"{model_name}.yaml")
    assert os.path.exists(config_path), f'Did not find generation configuration at {config_path}.'
    # read in the yaml file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    # load href from huggingface
    raw_text_prompts = []  # category -> list of example dicts
    raw_reference = []
    href_data = datasets.load_dataset(args.dataset)[args.split]
    for example in href_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        if example['generator_a'] == model_name or example['generator_b'] == model_name:
            raw_text_prompts.append(example['instruction'])
            raw_reference.append(example['reference'])

    # prepare the input and config
    if "openai" not in config: # local model
        tokenizer = load_hf_tokenizer(
            model_name_or_path=config['model_name_or_path'],
            tokenizer_name_or_path=config['tokenizer_name_or_path'],
            use_fast_tokenizer=not config['use_slow_tokenizer'],
        )
        if config['use_vllm']: # load vllm
            foundOpenAI = False
            while(not foundOpenAI):
                try:            
                    openai_client = OpenAI(
                        base_url="http://localhost:9898/v1",
                    )
                    foundOpenAI = True
                except:
                    wait_time = 120
                    logging.warning(f"Waiting for the server to start, waiting for {wait_time}s...")
                    wait(wait_time)
        else:
            model = load_hf_lm(
                model_name_or_path=config['model_name_or_path'],
                load_in_8bit=config['load_in_8bit'],
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=config['gptq'],
            )
            # modify tokenizer if required
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                logging.info("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

        # apply chat format
        if 'format' in config:
            prompts = []
            references = []
            for prompt, reference in zip(raw_text_prompts, raw_reference):
                if config['format'] == 'default':
                    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reference}]
                    prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False)
                    if reference not in prompt:
                        place_holder_messages = [{"role": "user", "content": "placeholder"}, {"role": "assistant", "content": reference}]
                        placeholder = create_prompt_with_huggingface_tokenizer_template(place_holder_messages, tokenizer, add_bos=False)
                        reference = find_shared_segment(prompt, placeholder).strip()
                    assert reference in prompt, [prompt, reference]
                    prompts.append(prompt)
                    references.append(reference)
                else:
                    prompts.append(config['format'].format(prompt=prompt))
                    references.append(reference)
        else:
            prompts = []
            references = []
            for prompt, reference in zip(raw_text_prompts, raw_reference):
                prompts.append(f"{prompt}\n{reference}")
                references.append(reference)
    else: # load openai model
        raise NotImplementedError("OpenAI not supported to generate the loss!")

    logging.info("Stats:")
    logging.info(f"{len(prompts)}")

    # config saving path
    save_dir = os.path.join(args.save_dir, model_name)
    save_path = os.path.join(save_dir, "perplexities.jsonl")
    if os.path.exists(save_path):
        return
    else:
        os.makedirs(save_dir, exist_ok=True)
    # generate outputs
    if config['model_name_or_path'] is not None: # local model
        if config['use_vllm']:
            # logging.info(f"Using VLLM:{sampling_params}")
            foundOpenAI = False
            while(not foundOpenAI):
                try:
                    vllm_outputs = openai_client.completions.create(
                        model=config['model_name_or_path'],
                        prompt=prompts,
                        max_tokens=0,
                        echo=True,
                        logprobs=1
                    ).choices
                    foundOpenAI = True
                except:
                    wait_time = 30
                    logging.warning(f"Waiting for the server to start, waiting for {wait_time}s...")
                    time.sleep(wait_time)
            # assert False, vllm_outputs[0]
            prompt_ids_references = []
            for i in range(0, len(references), config['batch_size']):
                prompt_ids_references.extend(tokenizer(references[i:i+config['batch_size']], padding="longest", 
                                    return_tensors="pt", truncation=True, add_special_tokens=False).input_ids.tolist())
            assert len(prompt_ids_references) == len(vllm_outputs)
            outputs = []
            for output, prompt_ids_reference in zip(vllm_outputs, prompt_ids_references):
                prompt_ids_reference = [str(id) for id in prompt_ids_reference]
                prompt_ids = [list(obj.keys())[0] for i, obj in enumerate(output.prompt_logprobs) if i > 0]
                start_idx, end_idx = find_sublist_indices(prompt_ids, prompt_ids_reference)
                # prompt_logprobs = output.prompt_logprobs
                # prompt_logs = []
                # for i in range(start_idx, end_idx, 1):
                #     if i == 0:
                #         continue
                #     id = prompt_ids[i]
                #     assert id in prompt_logprobs[i], [i, id, prompt_logprobs[i+1]] 
                #     prompt_logs.append(prompt_logprobs[i][id].logprob)
                prompt_logs = output.logprobs.token_logprobs[start_idx: end_idx]
                # assert False, prompt_logs
                outputs.append(math.exp(-sum(prompt_logs) / len(prompt_logs))) 
        else:
            outputs = generate_perplexities(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                responses=references,
                batch_size=config['batch_size'] if 'batch_size' in config else 1,
                max_length=config['max_new_tokens']
            )
    else: # openai model generation
        raise NotImplementedError("OpenAI not supported to generate the loss!")
            
    # save the outputs
    with open(save_path, "w") as fout:
        for prompt, output in zip(raw_text_prompts, outputs):
            example = {
                "reference": prompt,
                "output": output,
                "generator": f"{model_name}",
            }
            fout.write(json.dumps(example) + "\n")


def main():
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="The huggingface model name or the path to a local directory that contains the model to use for evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="alrope/test_human_agreement",
        help="The huggingface dataset name or the path to a local file to use for evaluation."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="The split to use in dataset."
    )
    parser.add_argument(
        "--nr_category",
        type=str,
        default=["Generation", "Open QA", "Brainstorm", "Rewrite", "Summarize",
                 "Classify", "Closed QA", "Extract", "Reasoning Over Numerical Data",
                 "Multi-Document Synthesis", "Fact Checking or Attributed QA"],
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
    # generation arguments
    parser.add_argument(
        "--generation_config_dir",
        type=str,
        default="href/generation/configs_mine-t=0.0",
        help="The directory to store downloaded datasets, models, and intermmediate annotation files.",
    )
    args = parser.parse_args()

    # set up
    random.seed(args.seed)
    os.environ['HF_HOME'] = args.cache_dir

    generate_loss(args)

if __name__ == "__main__":
    main()