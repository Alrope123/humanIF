import os
import json
import argparse
import logging
import random
from collections import defaultdict
import torch
import datasets
import vllm
from openai import OpenAI

from href.generation.utils import generate_completions, dynamic_import_function, load_hf_lm, load_hf_tokenizer


def generate(args):
    # model_name_or_path and openai_engine cannot be both None or both not None
    assert (args.model_name_or_path is not None) or (args.openai_engine is not None), "Either model_name_or_path or openai_engine should be specified."
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    # we skip everything if all outputs have been generate
    need_to_load_model = False
    for category in args.nr_category():
        model_name = (os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None \
            else args.openai_engine) + f"-t={args.temperature}" 
        save_path = os.path.join(args.save_dir, model_name, category.lower().replace(" ", "_"), "responses.jsonl")
        need_to_load_model = need_to_load_model or not os.path.exists(save_path)
    if not need_to_load_model:
        logging.info("Found all saved generations, reusing the generations.")
        return

    # load chat formatting
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    # load href from huggingface
    raw_text_prompts = defaultdict(list)  # category -> list of example dicts
    href_data = datasets.load_dataset(args.dataset)[args.split]
    for example in href_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        raw_text_prompts[category].append(example['instruction'])
    

    # prepare the input and config
    if args.model_name_or_path is not None: # local model
        # we always load the tokenizer for vllm or hf models
        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        if args.use_vllm: # load vllm
            vllm_model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                tensor_parallel_size=torch.cuda.device_count(),
                download_dir=f"{args.cache_dir}/models"
            )
            sampling_params = vllm.SamplingParams(
                temperature=args.temperature,  # greedy decoding
                max_tokens=args.max_new_tokens,
                repetition_penalty=1.0
            )
        else: # load hf model
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
            )
            # modify tokenizer if required
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                logging.info("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

        # apply chat format
        if args.use_chat_format:
            prompts = {}
            for category, category_prompts in raw_text_prompts.items():
                formatted_prompts = []
                for prompt in category_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                    formatted_prompts.append(prompt)
                prompts[category] = formatted_prompts
        else:
            prompts = dict(raw_text_prompts)
    else: # load openai model
        openai_client = OpenAI()
        prompts = dict(raw_text_prompts)


    logging.info("Stats:")
    for category, category_prompts in prompts.items():
        logging.info(f"{category}\t{len(category_prompts)}")

    # generate outputs
    for category, category_prompts in prompts.items():
        assert category in args.nr_category

        # config saving path
        model_name = (os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None \
            else args.openai_engine) + f"-t={args.temperature}" 
        save_dir = os.path.join(args.save_dir, model_name, category.lower().replace(" ", "_"))
        save_path = os.path.join(save_dir, "responses.jsonl")
        if os.path.exists(save_path):
            logging.info(f"Found saved generations for {category}, reusing the generations.")
            continue
        else:
            os.makedirs(save_dir, exist_ok=True)

        logging.info(f"Running inference on category: {category}")
        if args.model_name_or_path is not None: # local model
            if args.use_vllm:
                logging.info(f"Using VLLM:{sampling_params}")
                category_outputs = vllm_model.generate(category_prompts, sampling_params)
                category_outputs = [it.outputs[0].text for it in category_outputs]
            else:
                category_outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=category_prompts,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False if args.temperature == 0.0 else True,
                    temperature=args.temperature,
                    batch_size=args.batch_size if args.batch_size else 1,
                )
        else: # openai model generation
            assert not args.use_chat_format
            category_outputs = []
            for prompt in category_prompts:
                response = openai_client.chat.completions.create(
                    model=args.openai_engine,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                category_outputs.append(response.choices[0].message.content)
                
        # save the outputs
        with open(save_path, "w") as fout:
            for prompt, output in zip(raw_text_prompts[category], category_outputs):
                example = {
                    "instruction": prompt,
                    "output": output,
                    "generator": f"{model_name}-t={args.temperature}",
                    "dataset": f"href_{category}"
                }
                fout.write(json.dumps(example) + "\n")


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
        "--model_name_or_path",
        type=str,
        default=None,
        help="The huggingface model name or the path to a local directory that contains the model to use for evaluation.",
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="alrope/dev_test",
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
        "--response_dir",
        type=str, 
        default=None,
        help="The directory that contains pre-generated model outputs. If specified, we will skip output generation and jump directly into evaluation."
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
        help="The huggingface tokenizer name or the path to a local directory that contains the tokenizer to use for evaluation. If not specified, we will use the same ones as `model_name_or_path`.",
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
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature we use for model generation.",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for generation."
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
        default="href.generation.templates.create_prompt_with_huggingface_tokenizer_template", 
        help="The name of the function to use to create the chat format. This function will be dynamically imported. Functions are specified in generation/templates.py."
    )
    args = parser.parse_args()

    # set up
    random.seed(args.seed)
    os.environ['HF_HOME'] = args.cache_dir

    generate(args)

if __name__ == "__main__":
    main()