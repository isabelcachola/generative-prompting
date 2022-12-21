#!/bin/bash
#SBATCH -A mdredze1_gpu
#SBATCH --partition=a100
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time 3:00:00
#SBATCH --output /home/icachol1/genprompt.sample.out
#SBATCH --mail-user cachola.isabel@gmail.com
#SBATCH --mem=180GB
module load anaconda
conda activate genprompt
FILENAME=sample
cd /home/icachol1/generative-prompting/genprompt
python main.py \
-o $FILENAME.results \
--dataset_name out/$FILENAME.json \
--max_length 2048 \
--model_name gpt-j \
--batch_size 2