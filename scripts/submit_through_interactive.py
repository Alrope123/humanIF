import copy
import subprocess
import yaml
import random
import re
import itertools
from datetime import date
import argparse
import json
import os

# beaker session create --budget ai2/oe-adapt --cpus 15.5 --gpus 1 --workspace ai2/pradeepd-open-instruct --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env OPENAI_API_KEY=openai_api_key --secret-env HF_TOKEN=huggingface_token 

def parse_command(command):
    assert command.startswith("beaker session create")
    skipping_keys = ['no-update-default-image']
    
    output_kv = {}
    arguments = command.split()
    current_key = ""
    for arg in arguments:
        if arg.startswith('--'):
            if arg == '15.5':
                assert False
            arg = arg[len('--'):]
            if arg not in skipping_keys:
                current_key = arg
        elif current_key != "":
            if arg.startswith('beaker://'):
                arg = arg[len('beaker://'):]
            if "=" in arg:
                arg = (arg.split('=')[0], arg.split('=')[1])
            if type(arg) == str and arg.isnumeric():
                arg = int(arg)
            if current_key in output_kv:
                if type(output_kv[current_key]) != list:
                    output_kv[current_key] = [output_kv[current_key]]
                output_kv[current_key].append(arg)
            else:
                print([current_key, arg])
                output_kv[current_key] = arg
            current_key = ""
    return output_kv


def main(args):
    args = vars(args)

    today = date.today().strftime("%m%d%Y")
    with open("beaker_configs/default.yaml", 'r') as f:
        default_yaml = f.read()
    d = yaml.load(default_yaml, Loader=yaml.FullLoader)

    args.update(parse_command(args['command']))
    args['name'] = args['name'].replace('/', '_')
    del(args['command'])
    print(f"Submitting with the configurations:\n {json.dumps(args)}")

    d['description'] = args['name'] + "_" + today
    d['budget'] = args['budget']
    
    d['tasks'][0]['name'] = args['name'] + "_" + today
    d['tasks'][0]['image']['beaker'] = args['image']
    d['tasks'][0]['arguments'] = [args['argument']]
    
    if type(args['secret-env']) != list:
        args['secret-env'] = [args['secret-env']]
    if 'secret-env' in args:
        if 'envVars' not in d['tasks'][0]:
            d['tasks'][0]['envVars'] = []
        if 'secret-env' not in args:
            args['secret-env'] = []
        elif type(args['secret-env']) != list:
            args['secret-env'] = [args['secret-env']]
        for k_and_v in args['secret-env']:
            d['tasks'][0]['envVars'].append({
                "name": k_and_v[0],
                "secret": k_and_v[1]
            })

    if 'mount' in args:
        if 'datasets' not in d['tasks'][0]:
            d['tasks'][0]['datasets'] = []
        if 'datasets' not in args:
            args['datasets'] = []
        elif type(args['datasets']) != list:
            args['datasets'] = [args['datasets']]
    
        if type(args['mount']) != list:
            args['mount'] = [args['mount']]
        for k_and_v in args['mount']:
            if  k_and_v[0].startswith("/"):
                d['tasks'][0]['datasets'].append({
                    "mountPath": k_and_v[1],
                    "source": {"hostPath": k_and_v[0]}
                })
            else:
                d['tasks'][0]['datasets'].append({
                    "mountPath": k_and_v[1],
                    "source": {"beaker": k_and_v[0]}
                })
    
    if 'gpus' in args:
        d['tasks'][0]['resources']['gpuCount'] = args['gpus']
    if 'cpus' in args:
        d['tasks'][0]['resources']['cpuCount'] = args['cpus']
    if 'cluster' in args:
        d['tasks'][0]['constraints']['cluster'] = args['cluster']
    if 'priority' in args:
        d['tasks'][0]['context']['priority'] = args['priority']
    d['tasks'][0]['context']['preemptible'] = True

    fn = "beaker_configs/auto_created/{}.yaml".format(args['name'])
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=False)
    file.close()


    cmd = "beaker experiment create {} --workspace {}".format(fn, args['workspace'])
    subprocess.Popen(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
    )
    parser.add_argument(
        "--argument",
        type=str,
    )
    parser.add_argument(
        "--command",
        type=str,
    )
    parser.add_argument(
        "--priority",
        type=str,
        default="normal"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        nargs="+",
        default=["ai2/allennlp-cirrascale", "ai2/jupiter-cirrascale-2", "ai2/pluto-cirrascale"]
    )
    
    args = parser.parse_args()

    main(args)
