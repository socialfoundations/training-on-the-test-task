#!/bin/bash

VENV_PATH="/home/rolmedo/axo121/"

source "$VENV_PATH/bin/activate"
module load cuda/12.1

lm_eval --model hf --model_args $1 --tasks $2 --batch_size 1 --log_samples --output_path $3 ${@:5}

rm -rf $4