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

def init_dataset(dataset_name: str) -> List[str]:
    if dataset_name == 'dummy':
        dataset = pd.read_csv('triviaqa.sample', header=None)
        examples = dataset[0].values.tolist()
        return examples
    else:
        examples = json.load(open('dataset_name'))
        return list(examples.values())

def save_generations(results, fout_path):
    with open(fout_path, 'w') as fout:
        for r in results:
            fout.write(results.replace('\n', ' ') + '\n')

def main(args):
    model = init_model(args.model_name, args.max_length)
    examples = init_dataset(args.dataset_name)
    output = model.generate_from_prompts(examples)
    save_generations(output, args.outfile)
    print(output)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', '-o', dest='outfile', required=True)
    parser.add_argument('--model_name', default='gpt-neo', choices=['gpt-neo', 'gpt-j', 'lead3'])
    parser.add_argument('--dataset_name', default='dummy', choices=['dummy', 'tedtalk-sample'])
    parser.add_argument('--max_length', default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')