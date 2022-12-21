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


python generate.py --batch $BATCH --model EleutherAI/gpt-j-6B --input_file out_xx/k$k/baseline+ted_dev_en-$tgt.raw.en.json > results/en-$tgt/baseline_k$k.en-$tgt.results

python generate.py --batch $BATCH --model EleutherAI/gpt-j-6B --input_file out_xx/k$k/random+ted_dev_en-$tgt.raw.en.json > results/en-$tgt/random_k$k.en-$tgt.results