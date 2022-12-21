'''
Loads and processes each dataset 
'''
import argparse
import numpy as np
from typing import Optional
import os
from tqdm import tqdm
import json
import random

from datasets import load_dataset


templates = {
    "cnndm": 
     {
        'zeroshot': "{}\nSummarize the above article in 2 sentences.",
        'fewshot': "Article: {}\n Summary: {}"
    }
}

    

def make_filenames(args):
    filename = ''
    if args.zeroshot:
        filename = 'zeroshot'
    elif args.random:
        filename = f'random_k={args.k}'
    else:
        filename= f'baseline_k={args.k}'
    
    filename += '+'
    filename += 'cnndm'

    return os.path.join(args.outdir, filename + '.json'), os.path.join(args.outdir, 'cnndm.answers')

def sample_k(dataset, k):
    n = len(dataset)
    sampled_idxs = random.sample(range(n), k)
    sampled = dataset[sampled_idxs]
    return sampled

def get_context_from_example(ex, max_context):
    context = ex['article']
    context = ' '.join(context.split()[:max_context])
    return context

def get_context_from_dict(ex, idx, max_context):
    context = ex['article'][idx]
    context = ' '.join(context.split()[:max_context])
    return context

def main(args):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split='test')

    prompts_file, answers_file = make_filenames(args)
    examples = {}
    answers = []
    if args.zeroshot:
        for i, ex in tqdm(enumerate(dataset)):
            article = get_context_from_example(ex, args.max_context)
            answer = ex['highlights']
            answers.append(answer.replace('\n', ' ').strip())
            formatted_ex = templates['cnndm']['zeroshot'].format(article)
            formatted_ex = formatted_ex.split(' ')
            if len(formatted_ex) > args.max_total:
                formatted_ex = formatted_ex[len(formatted_ex)-args.max_total:]
            formatted_ex = ' '.join(formatted_ex)
            examples[f'{i}'] = formatted_ex

    elif args.random:
        train_dataset = load_dataset("cnn_dailymail", "3.0.0", split='train')
        for i, ex in tqdm(enumerate(dataset)):
            # Demonstrations
            demonstrations = sample_k(train_dataset, args.k*2)
            formatted_ex = ''
            for dem_idx in range(args.k):
                article = get_context_from_dict(demonstrations, dem_idx, args.max_context)
                answer = demonstrations['highlights'][dem_idx+args.k]
                formatted_demo = templates['cnndm']['fewshot'].format(article, answer)
                formatted_ex += formatted_demo

            # Validation question
            article = get_context_from_example(ex, args.max_context)
            answer = ex['highlights']
            answers.append(answer.replace('\n', ' ').strip())
            formatted_ex += templates['cnndm']['fewshot'].format(article, '')
            formatted_ex = formatted_ex.split(' ')
            if len(formatted_ex) > args.max_total:
                formatted_ex = formatted_ex[len(formatted_ex)-args.max_total:]
            formatted_ex = ' '.join(formatted_ex)
            examples[f'{i}'] = formatted_ex
    else:
        train_dataset = load_dataset("cnn_dailymail", "3.0.0", split='train')
        for i, ex in tqdm(enumerate(dataset)):
            # Demonstrations
            demonstrations = sample_k(train_dataset, args.k)
            formatted_ex = ''
            for dem_idx in range(args.k):
                article = get_context_from_dict(demonstrations, dem_idx, args.max_context)
                answer = demonstrations['highlights'][dem_idx]
                formatted_demo = templates['cnndm']['fewshot'].format(article, answer)
                formatted_ex += formatted_demo + '\n'

            # Validation question
            article = get_context_from_example(ex, args.max_context)
            answer = ex['highlights']
            answers.append(answer.replace('\n', ' ').strip())
            formatted_ex += templates['cnndm']['fewshot'].format(article, '')
            formatted_ex = formatted_ex.split(' ')
            if len(formatted_ex) > args.max_total:
                formatted_ex = formatted_ex[len(formatted_ex)-args.max_total:]
            formatted_ex = ' '.join(formatted_ex)
            examples[f'{i}'] = formatted_ex
    
    with open(prompts_file, 'w') as fout:
        json.dump(examples, fout, indent=4)

    with open(answers_file, 'w') as fout:
        fout.write("\n".join(answers))
    
            

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--zeroshot", default=False, action="store_true")
    parser.add_argument("--random", default=False, action="store_true")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--outdir", default='../cnndm_data')
    parser.add_argument("--max_context", type=int, default=256)
    parser.add_argument("--max_total", type=int, default=2000)

    args = parser.parse_args()
    main(args)