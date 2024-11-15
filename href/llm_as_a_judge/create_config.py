import os
import json
import argparse
import yaml
import random
import copy

def remove_substring_between(original_string, start_substring, end_substring):
    while True:
        # Find the start and end positions of the substrings
        start_pos = original_string.find(start_substring)
        end_pos = original_string.find(end_substring, start_pos + len(start_substring))
        
        # If either start or end substring is not found, break the loop
        if start_pos == -1 or end_pos == -1:
            break
        
        # Remove the substrings and the text in between
        original_string = original_string[:start_pos] + original_string[end_pos + len(end_substring):]
    
    return original_string


def create_config(args):
    random.seed(42)

    model = args.model_config_name
    template_name = args.template_name
    config_dir = args.config_dir
    no_example = args.no_example
    temperature = args.temperature
    do_sampling = temperature > 0.0
    
    # load config template
    root_directory = os.path.join("href", "llm_as_a_judge")
    with open(os.path.join(root_directory, "config_template.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    # load config setting
    model_configs = json.load(open(os.path.join(root_directory, "model_settings.json"), 'r'))[model]
    # load selected template
    with open(os.path.join(root_directory, "prompt_templates", f"{template_name}.txt"), 'r') as f:
        template = f.read()
    
    # modify the config name
    template_name = model  + "_" + template_name + ("" if not no_example else "_no_ex") + ("" if not do_sampling else "_sample")
    config[template_name] = copy.deepcopy(config["base"])
    del config["base"]
    config[template_name]["prompt_template"] = os.path.join(template_name, "prompt.txt")
    
    # change the default setting from the config template using the config setting file
    for k, v in model_configs.items():
        if k == "template_kwargs":
            for place_holder, text in v.items():
                template = template.replace("{" + place_holder + "}", text) 
        else:
            config[template_name][k] = v
    
    # do sampling while judging
    if do_sampling and model not in ["gpt4", "gpt4-turbo"]:
        config[template_name]["completions_kwargs"]["do_sample"] = True
    elif not do_sampling and model not in ["gpt4", "gpt4-turbo"]:
        config[template_name]["completions_kwargs"]["do_sample"] = False
    config[template_name]["completions_kwargs"]["temperature"] = temperature


    # whether to exclude demonstrations
    if no_example:
        template = remove_substring_between(template, "{" + "example_begin" + "}", "{" + "example_end" + "}")
    else:
        template = template.replace("{" + "example_begin" + "}", "").replace("{" + "example_end" + "}", "")

    # save the resulting configuration
    if not os.path.exists(os.path.join(config_dir, template_name)):
        os.mkdir(os.path.join(config_dir, template_name))
    with open(os.path.join(config_dir, template_name, "prompt.txt"), 'w') as f:
        f.write(template)
    with open(os.path.join(config_dir, template_name, "configs.yaml"), 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_config_name",
        type=str,
        required=True,
        help="The name of the model configuration used as the judge defined in href/llm_as_a_judge/model_settings.json."
    )

    parser.add_argument(
        "--template_name",
        type=str,
        required=True,
        help="The name of the template file in `href/llm_as_a_judge/prompt_templates` (without the suffix)."
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="href/llm_as_a_judge/configs",
        help="the directory to save the resulting configuration."
    )

    parser.add_argument(
        "--no_example",
        default=False,
        action="store_true",
        help="If specified, remove the demonstration examples in the prompt."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for the judge model."
    )

    args = parser.parse_args()
    create_config(args)


if __name__ == "__main__":
    main()