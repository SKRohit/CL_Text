#!/bin/bash

python augment_data.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-de \
    --rev_model_name_or_path Helsinki-NLP/opus-mt-de-en \
    --infer_file ../data/custom_data.csv \
    --batch_size 32 \
    --max_length 256 \
    --text_col text \
    --num_beams 4 \
    "$@"
