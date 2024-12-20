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

from href.generation.utils import generate_perplexities, create_prompt_with_huggingface_tokenizer_template, load_hf_lm, load_hf_tokenizer

def generate_loss(args):
    assert args.model_name is not None, "Model name should be specified."
    model_name = args.model_name
    config_path = os.path.join(args.generation_config_dir, f"{model_name}.yaml")
    assert os.path.exists(config_path), 'Did not find generation configuration.'
    # read in the yaml file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # we skip everything if all outputs have been generate
    need_to_load_model = False
    for category in args.nr_category:
        save_path = os.path.join(args.save_dir, model_name, category.lower().replace(" ", "_"), "responses.jsonl")
        need_to_load_model = need_to_load_model or not os.path.exists(save_path)
    if not need_to_load_model:
        logging.info("Found all saved generations, reusing the generations.")
        return


    # load href from huggingface
    raw_text_prompts = defaultdict(list)  # category -> list of example dicts
    raw_reference = defaultdict(list)
    href_data = datasets.load_dataset(args.dataset)[args.split]
    for example in href_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        raw_text_prompts[category].append(example['instruction'])
        raw_reference[category].append(example['reference'])

    # prepare the input and config
    if "openai" not in config: # local model
        tokenizer = load_hf_tokenizer(
            model_name_or_path=config['model_name_or_path'],
            tokenizer_name_or_path=config['tokenizer_name_or_path'],
            use_fast_tokenizer=not config['use_slow_tokenizer'],
        )
        if config['use_vllm']: # load vllm
            vllm_model = vllm.LLM(
                model=config['model_name_or_path'],
                tokenizer=config['tokenizer_name_or_path'] if config['tokenizer_name_or_path'] is not None else config['model_name_or_path'],
                tensor_parallel_size=torch.cuda.device_count(),
                download_dir=f"{args.cache_dir}/models"
            )
            sampling_params = vllm.SamplingParams(
                temperature=config['temperature'],  # greedy decoding
                max_tokens=1,
                prompt_logprobs=1
            )
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
            prompts = {}
            references = {}
            for category, category_prompts in raw_text_prompts.items():
                category_reference = raw_reference[category]
                formatted_prompts = []
                formatted_reference = []
                for prompt, reference in zip(category_prompts, category_reference):
                    if config['format'] == 'default':
                        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reference}]
                        prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False)
                        messages = [{"role": "assistant", "content": reference}]
                        reference = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False)
                        formatted_reference.append(reference)
                        assert prompt.endswith(reference), [prompt, reference]
                    else:
                        formatted_prompts.append(config['format'].format(prompt) + reference)
                        formatted_reference.append(reference)
                prompts[category] = formatted_prompts
                references[category] = formatted_reference
        else:
            prompts = {}
            references = {}
            for category, category_prompts in raw_text_prompts.items(): 
                category_reference = raw_reference[category]
                formatted_prompts = []
                formatted_reference = []
                for prompt, reference in zip(category_prompts, category_reference):
                    formatted_prompts.append(f"{prompt}\n{reference}")
                    formatted_reference.append(reference)
                prompts[category] = formatted_prompts
                references[category] = formatted_reference
                
    else: # load openai model
        raise NotImplementedError("OpenAI not supported to generate the loss!")

    logging.info("Stats:")
    for category, category_prompts in prompts.items():
        logging.info(f"{category}\t{len(category_prompts)}")

    # generate outputs
    for category, category_prompts in prompts.items():
        assert category in args.nr_category
        category_reference = references[category]

        # config saving path
        save_dir = os.path.join(args.save_dir, model_name, category.lower().replace(" ", "_"))
        save_path = os.path.join(save_dir, "perplexities.jsonl")
        if os.path.exists(save_path):
            logging.info(f"Found saved generations for {category}, reusing the generations.")
            continue
        else:
            os.makedirs(save_dir, exist_ok=True)

        logging.info(f"Running inference on category: {category}")
        if config['model_name_or_path'] is not None: # local model
            if config['use_vllm']:
                logging.info(f"Using VLLM:{sampling_params}")
                category_outputs = vllm_model.generate(category_prompts, sampling_params)
                assert False, category_outputs[0]
                category_outputs = [it.outputs[0].text for it in category_outputs]
            else:
                category_outputs = generate_perplexities(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=category_prompts,
                    responses=category_reference,
                    batch_size=config['batch_size'] if 'batch_size' in config else 1,
                    max_length=config['max_new_tokens']
                )
        else: # openai model generation
            raise NotImplementedError("OpenAI not supported to generate the loss!")
                
        # save the outputs
        with open(save_path, "w") as fout:
            for prompt, output in zip(raw_text_prompts[category], category_outputs):
                example = {
                    "reference": prompt,
                    "output": output,
                    "generator": f"{model_name}",
                    "dataset": f"href_{category}"
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
    # generation arguments
    parser.add_argument(
        "--generation_config_dir",
        type=str,
        default="configs_mine-t=0.0",
        help="The directory to store downloaded datasets, models, and intermmediate annotation files.",
    )
    args = parser.parse_args()

    # set up
    random.seed(args.seed)
    os.environ['HF_HOME'] = args.cache_dir

    generate_loss(args)

if __name__ == "__main__":
    main()