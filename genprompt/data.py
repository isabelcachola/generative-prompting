'''
Loads and processes each dataset 
'''
import argparse
import numpy as np
from typing import Optional
import os
from tqdm import tqdm
import json

templates = {
    "TED": {
        "zero": "What is the {} translation of {} A: {}",
        "few": "{} = {}"
    }
}
    
def apply_template(template, context: str, prompt: bool, answer: Optional[str]='', language: Optional[str]=None):
    applied = None
    #breakpoint()
    if language:
        applied = template.format(language, context, answer)
    else:
        applied = template.format(context, answer)
    if prompt:
        applied = applied[:-1] #remove trailing space
    return applied
    # prompt_text = apply_template(templates['TED']['zero'], line.strip(), prompt, language=args.tgt_lang)

def retrieve_examples(context_file, answer_file, k: int, total_examples: int, random: bool):
    items = np.random.choice(range(total_examples), k, replace=False)
    examples = None
    with open(context_file) as context_f, open(answer_file) as answer_f:
        line_pointer = context_f.readlines()
        template = templates['TED']['few']
        contexts = [line_pointer[i] for i in items]
        line_pointer = answer_f.readlines()
        if random:
            new_items = np.random.choice(range(total_examples), k, replace=False)
            answers = [line_pointer[i] for i in new_items]
        else:
            answers = [line_pointer[i] for i in items]
        examples = [apply_template(template, context.strip(), False, answer.strip()) for (context, answer) in zip(contexts, answers)]
    return examples
    

def make_filename(args):
    filename = ''
    if args.zeroshot:
        filename = 'zeroshot'
    elif args.random:
        filename = 'random'
    else:
        filename= 'baseline'
    
    filename += '+'
    filename += os.path.split(args.test_context_file)[1]
    filename += '.json'

    return filename

def main(args):
    if not args.zeroshot:
        total_examples = None
        with open(args.context_file) as f:
            total_examples = len(f.readlines())
    filename = make_filename(args)
    with open(os.path.join(args.output_dir,filename), 'w+') as out_f, open(args.test_context_file) as in_f:
        output_dict = {}
        for i, line in tqdm(enumerate(in_f.readlines())):
            if args.zeroshot:
                prompt=True
                prompt_text = apply_template(templates['TED']['zero'], line.strip(), prompt, language=args.tgt_lang)
                output_dict[i] = (prompt_text + '\n')
            else:
                prompt_text = apply_template(templates['TED']['few'], line.strip(), True)
                if args.random:
                   examples = retrieve_examples(args.context_file, args.answer_file, args.k, total_examples, random=True)
                else:
                    examples = retrieve_examples(args.context_file, args.answer_file, args.k, total_examples, random=False)
                example_string = '\n'.join(examples)
                output_dict[i] = example_string + '\n' + prompt_text + '\n'
        json.dump(output_dict, out_f, indent=4)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--zeroshot", default=False, action="store_true")
    parser.add_argument("--random", default=False, action="store_true")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--context_file", type=str, required=False)
    parser.add_argument("--answer_file", type=str, required=False)
    parser.add_argument("--test_context_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=False) # if zero shot

    args = parser.parse_args()
    main(args)