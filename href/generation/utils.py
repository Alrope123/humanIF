import torch
import torch.nn.functional as F
import tqdm
import os
from importlib import import_module
from transformers import StoppingCriteria, pipeline


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)
    
    
@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True, disable_tqdm=False, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
        
            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.no_grad()
def generate_perplexities(model, tokenizer, prompts, responses, batch_size=1, add_special_tokens=True, disable_tqdm=False, **kwargs):
    perplexities = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Perplexity")

    num_return_sequences = kwargs.get("num_return_sequences", 1)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                        truncation=True, add_special_tokens=add_special_tokens, **kwargs)
        tokenized_responses = tokenizer(batch_responses, padding="longest", return_tensors="pt",
                                        truncation=True, add_special_tokens=add_special_tokens, **kwargs)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        attention_mask_response = tokenized_responses.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
            )
            batch_logits = batch_outputs.logits

            # Calculate loss for each prompt in the batch
            batch_perplexities = []
            for idx, prompt in enumerate(batch_prompts):
                prompt_length = (attention_mask[idx] == 1).sum()  # Length of prompt in tokens
                response_length = (attention_mask[idx] == 1).sum()
                response_begin_idx = prompt_length - response_length
                logits_for_prompt = batch_logits[response_begin_idx, :prompt_length - 1]  # Ignore last token for perplexity calculation
                batch_target_ids = batch_input_ids[response_begin_idx, 1:prompt_length]

                # Cross entropy loss between model predictions and actual next tokens
                loss = F.cross_entropy(logits_for_prompt.view(-1, logits_for_prompt.size(-1)), batch_target_ids, reduction='mean')
                perplexity = torch.exp(loss).item()
                batch_perplexities.append(perplexity)

        except Exception as e:
            print("Error when generating perplexity for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_perplexities = [""] * len(batch_prompts) * num_return_sequences

        perplexities += batch_perplexities

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(perplexities) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return perplexities


def load_hf_lm(
        model_name_or_path, 
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        gptq_model=False,
        token=os.getenv("HF_TOKEN", None),
    ):

    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
    trust_remote_code = True
    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True, trust_remote_code=trust_remote_code
        )
        model = model_wrapper.model  
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True,
            token=token,
            trust_remote_code=trust_remote_code
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
            )
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()
    return model

def load_hf_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        use_fast_tokenizer=True,
        padding_side="left",
        token=os.getenv("HF_TOKEN", None),
    ):
        from transformers import AutoTokenizer

        trust_remote_code = True

        if not tokenizer_name_or_path:
            tokenizer_name_or_path = model_name_or_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, token=token, trust_remote_code=trust_remote_code)
        except:
            # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token, trust_remote_code=trust_remote_code)
        # set padding side to left for batch generation
        tokenizer.padding_side = padding_side
        # set pad token to eos token if pad token is not set (as is the case for llama models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer


def create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False, add_generation_prompt=True):
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, 
        add_generation_prompt=add_generation_prompt)
    if add_bos:
        formatted_text = tokenizer.bos_token + formatted_text
    return formatted_text


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
 