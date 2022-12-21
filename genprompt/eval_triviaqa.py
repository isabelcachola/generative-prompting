import argparse
import numpy as np
from typing import Optional
import os
from tqdm import tqdm
import string, re
import numpy as np


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_includes(prediction, truth):
    return int(normalize_text(prediction) in normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)


def main(args):
    em = []
    f1 = []
    includes = []
    predictions = open(args.preds).readlines()
    references = open(args.ref).readlines()
    for pred, ref in zip(predictions, references):
        gold_answers = ref.split(';')
        em_score = max((compute_exact_match(pred, answer)) for answer in gold_answers)
        f1_score = max((compute_f1(pred, answer)) for answer in gold_answers)
        includes_score = max((compute_includes(pred, answer)) for answer in gold_answers)
        em.append(em_score)
        f1.append(f1_score)
        includes.append(includes_score)
    print(f'F1 = {np.mean(f1)}')
    print(f'EM = {np.mean(em)}')
    print(f'Inclusive EM = {np.mean(includes)}')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("preds")
    parser.add_argument("ref")

    args = parser.parse_args()
    main(args)