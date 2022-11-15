'''
Main script
'''
import argparse
import logging
import time
import torch
import pandas as pd
from genprompt import data, models
import json
import ijson
from typing import List

def init_model(model_name, max_length):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device {device}")
    if model_name == 'lead3':
        return models.Lead3()
    elif model_name == 'gpt-neo':
        return models.GPT(max_length, 'EleutherAI/gpt-neo-1.3B', device)
    elif model_name == 'gpt-j':
        return models.GPT(max_length, 'EleutherAI/gpt-j-6B', device)
    else:
        raise ValueError()

def lazy_loader_json(fname):
    with open(fname, "rb") as f:
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if value and len(value) > 10:
                yield {'text': value}

def init_dataset(dataset_name: str) -> List[str]:
    if dataset_name == 'dummy':
        dataset = pd.read_csv('triviaqa.sample', header=None)
        examples = dataset[0].values.tolist()
        return examples
    else:
        # examples = lazy_loader_json(dataset_name)
        examples ={'text': list(json.load(open(dataset_name)).values())}
        return examples

def save_generations(results, fout_path):
    with open(fout_path, 'w') as fout:
        for r in results:
            fout.write(results.replace('\n', ' ') + '\n')

def main(args):
    logging.info(f'Loading dataset {args.dataset_name}')
    examples = init_dataset(args.dataset_name)

    logging.info(f'Loading model {args.model_name}')
    model_start_load_time = time.time()
    model = init_model(args.model_name, args.max_length)
    model_end_load_time = time.time()
    logging.info(f'Time to load model: {model_end_load_time-model_start_load_time} secs')
    

    logging.info(f'Running generations...')
    model.generate_from_prompts(examples, args.outfile)
    # save_generations(output, args.outfile)
    # print(output)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', '-o', dest='outfile', required=True)
    parser.add_argument('--dataset_name', '-d', dest='dataset_name', required=True)
    parser.add_argument('--model_name', default='gpt-neo', choices=['gpt-neo', 'gpt-j', 'lead3'])
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, 
                        format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')