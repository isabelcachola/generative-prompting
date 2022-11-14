#!/bin/bash

k=$1

# # make zero shot for each lang
# for lang in de id zh; do
#     python data.py --zeroshot --test_context_file ../data/en-$lang/ted_dev_en-$lang.raw.$lang --output_dir out --tgt_lang English
# done

# make baseline
for lang in de id zh; do
    python data.py --context_file ../data/en-$lang/context_samples_en-$lang.$lang  --answer_file ../data/en-$lang/context_samples_en-$lang.en  --k $k --test_context_file ../data/en-$lang/ted_dev_en-$lang.raw.$lang --output_dir out_k$k
done

# make random
for lang in de id zh; do
    python data.py --context_file ../data/en-$lang/context_samples_en-$lang.$lang --answer_file ../data/en-$lang/context_samples_en-$lang.en  --k $k --test_context_file ../data/en-$lang/ted_dev_en-$lang.raw.$lang --output_dir out_k$k --random
done