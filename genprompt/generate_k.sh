#!/bin/bash

#$ -l gpu=1,h_rt=03:00:00
#$ -q gpu.q@@v100
#$ -j y
#$ -o qlog/
#$ -cwd

source ~/.bashrc
conda deactivate && conda activate py38
module load cuda11.2/toolkit


BATCH=$1
tgt=$2
k=$3

mkdir -p results/$tgt-en/

python generate.py --batch $BATCH --model EleutherAI/gpt-j-6B --input_file out_k$k/baseline+ted_dev_en-$tgt.raw.$tgt.json > results/$tgt-en/baseline_k$k.$tgt-en.results

python generate.py --batch $BATCH --model EleutherAI/gpt-j-6B --input_file out_k$k/random+ted_dev_en-$tgt.raw.$tgt.json > results/$tgt-en/random_k$k.$tgt-en.results