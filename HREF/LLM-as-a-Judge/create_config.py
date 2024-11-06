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


def main(args):
    random.seed(42)

    model = args.model
    template_name = args.template_name
    config_dir = args.config_dir
    no_example = args.no_example
    do_sampling = args.do_sampling
    
    # load config template
    root_directory = os.path.join("HREF", "LLM-as-a-Judge")
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
        if type(v) == str:
            template = template.replace("{" + k + "}", v) 
        if k in ["fn_completions", "completions_kwargs"]:
            config[template_name][k] = v
    
    # do sampling while judging
    if do_sampling:
        if model not in ["gpt4", "gpt4-turbo"]:
            config[template_name]["completions_kwargs"]["do_sample"] = True
            config[template_name]["completions_kwargs"]["temperature"] = 0.9
        else:
            config[template_name]["completions_kwargs"]["temperature"] = 0.9

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="gpt4",
    )

    parser.add_argument(
        "--template_name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="HREF/LLM-as-a-Judge/configs",
    )

    parser.add_argument(
        "--no_example",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--do_sampling",
        default=False,
        action="store_true",
    )


    args = parser.parse_args()
    main(args)
