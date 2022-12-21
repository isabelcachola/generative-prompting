#!bin/bash

# CNN DM 
# python main.py \
# -o results/gpt3.zeroshot.cnndm.preds \
# -d cnndm_data/zeroshot+cnndm.json \
# --model_name gpt-3

# python main.py \
# -o results/gpt3.baseline_k=1.cnndm.preds \
# -d cnndm_data/baseline_k=1+cnndm.json \
# --model_name gpt-3

# python main.py \
# -o results/gpt3.random_k=1.cnndm.preds \
# -d cnndm_data/random_k=1+cnndm.json \
# --model_name gpt-3

# TriviaQA
# python main.py \
# -o results/gpt3.zeroshot.triviaqa.preds \
# -d triviaqa_data/zeroshot+triviaqa.json \
# --model_name gpt-3

python main.py \
-o results/gpt3.random_k=1.triviaqa.preds \
-d triviaqa_data/random_k=1+triviaqa.json \
--model_name gpt-3

python main.py \
-o results/gpt3.baseline_k=1.triviaqa.preds \
-d triviaqa_data/baseline_k=1+triviaqa.json \
--model_name gpt-3