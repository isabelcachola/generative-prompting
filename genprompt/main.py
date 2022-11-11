'''
Main script
'''
import argparse
import logging
import time
import pandas as pd
from genprompt import data, models
from typing import List

def init_model(model_name, max_length):
    if model_name == 'lead3':
        return models.Lead3()
    if model_name == 'gpt-neo':
        return models.GPT(max_length, model_name='EleutherAI/gpt-neo-1.3B')
    else:
        raise ValueError()

def init_dataset(dataset_name: str) -> List[str]:
    if dataset_name == 'dummy':
        dataset = pd.read_csv('triviaqa.sample', header=None)
        examples = dataset[0].values.tolist()
        return examples
    else:
        raise ValueError()

def main(args):
    model = init_model(args.model_name, args.max_length)
    examples = init_dataset(args.dataset_name)
    output = model.generate_from_prompts(examples)
    print(output)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt-neo', choices=['gpt-neo', 'lead3'])
    parser.add_argument('--dataset_name', default='dummy', choices=['dummy'])
    parser.add_argument('--max_length', default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')