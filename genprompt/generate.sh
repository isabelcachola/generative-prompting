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

mkdir -p results/$tgt-en/

python generate.py --batch $BATCH --model EleutherAI/gpt-j-6B --input_file out/zeroshot/zeroshot+ted_dev_en-$tgt.raw.$tgt.json > results/$tgt-en/zeroshot.$tgt-en.results
