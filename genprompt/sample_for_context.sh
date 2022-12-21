#!/bin/bash

# this script samples a number of data pairs from ted data in order to create few shot learning examples later. Stored in data folder 

data_dir=/exp/nverma/domain_adapt/raw/multitarget-ted/


for tgt in de id zh; do
    python sample.py $data_dir/en-$tgt/raw/ted_train_en-$tgt.raw.en $data_dir/en-$tgt/raw/ted_train_en-$tgt.raw.$tgt 32000 ../data/en-$tgt/context_samples_en-$tgt
done
