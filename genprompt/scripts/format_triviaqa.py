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

SKIP_THESE = [870, 1279, 5232]
templates = {
    "triviaqa": 
     {
        'zeroshot': "{}\n\nQ: {}\n\nA:",
        'fewshot': "{}\n\nQ: {}\n\nA: {}\n\n"
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
    filename += 'triviaqa'

    return os.path.join(args.outdir, filename + '.json'), os.path.join(args.outdir, 'triviaqa.answers')

def sample_k(dataset, k):
    n = len(dataset)
    sampled_idxs = random.sample(range(n), k)
    sampled = dataset[sampled_idxs]
    return sampled

def get_context_from_example(ex, max_context):
    context = ex['entity_pages']['wiki_context']
    if context == []:
        context = ex['search_results']['search_context'][0]
    else:
        context = context[0]
    context = ' '.join(context.split()[:max_context])
    return context

def get_context_from_dict(ex, idx, max_context):
    context = ex['entity_pages'][idx]['wiki_context']
    if context == []:
        context = ex['search_results'][idx]['search_context']
    if context != []:
        context = context[0]
    else:
        context = ''
    context = ' '.join(context.split()[:max_context])
    return context

def truncate_example(ex, i):
    formatted_ex = ex.split(' ')
    if len(formatted_ex) > args.max_total:
        formatted_ex = formatted_ex[len(formatted_ex)-args.max_total:]
    if i in SKIP_THESE:
        return ''
    formatted_ex = ' '.join(formatted_ex)
    return formatted_ex

def main(args):
    dataset = load_dataset("trivia_qa", "rc", split='validation')
    context = dataset['entity_pages'][0]['wiki_context']
    question = dataset['question'][0]
    answer = dataset['answer'][0]['value']
    prompts_file, answers_file = make_filenames(args)
    examples = {}
    answers = []
    if args.zeroshot:
        for i, ex in tqdm(enumerate(dataset)):
            context = get_context_from_example(ex, args.max_context)
            question = ex['question']
            answer = ";".join(ex['answer']['aliases'])
            formatted_ex = templates['triviaqa']['zeroshot'].format(context, question)
            formatted_ex = truncate_example(formatted_ex, i)
            if formatted_ex != '':
                answers.append(answer)
                examples[f'{i}'] = formatted_ex

    elif args.random:
        train_dataset = load_dataset("trivia_qa", "rc", split='train')
        for i, ex in tqdm(enumerate(dataset)):
            # Demonstrations
            demonstrations = sample_k(train_dataset, args.k*2)
            formatted_ex = ''
            for dem_idx in range(args.k):
                context = get_context_from_dict(demonstrations, dem_idx, args.max_context)
                question = demonstrations['question'][dem_idx]
                answer = demonstrations['answer'][dem_idx+args.k]['value']
                formatted_demo = templates['triviaqa']['fewshot'].format(context, question, answer)
                formatted_ex += formatted_demo

            # Validation question
            context = get_context_from_example(ex, args.max_context)
            question = ex['question']
            answer = ";".join(ex['answer']['aliases'])
            formatted_ex += templates['triviaqa']['zeroshot'].format(context, question)
            formatted_ex = truncate_example(formatted_ex, i)
            if formatted_ex != '':
                answers.append(answer)
                examples[f'{i}'] = formatted_ex
    else:
        train_dataset = load_dataset("trivia_qa", "rc", split='train')
        for i, ex in tqdm(enumerate(dataset)):
            # Demonstrations
            demonstrations = sample_k(train_dataset, args.k)
            formatted_ex = ''
            for dem_idx in range(args.k):
                context = get_context_from_dict(demonstrations, dem_idx, args.max_context)
                question = demonstrations['question'][dem_idx]
                answer = demonstrations['answer'][dem_idx]['value']
                formatted_demo = templates['triviaqa']['fewshot'].format(context, question, answer)
                formatted_ex += formatted_demo

            # Validation question
            context = get_context_from_example(ex, args.max_context)
            question = ex['question']
            answer = ";".join(ex['answer']['aliases'])
            formatted_ex += templates['triviaqa']['zeroshot'].format(context, question)
            formatted_ex = truncate_example(formatted_ex, i)
            if formatted_ex != '':
                answers.append(answer)
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
    parser.add_argument("--outdir", default='../triviaqa_data')
    parser.add_argument("--max_context", type=int, default=256)
    parser.add_argument("--max_total", type=int, default=2000)

    args = parser.parse_args()
    main(args)