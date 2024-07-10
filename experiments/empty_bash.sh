#!/bin/bash

VENV_PATH="/home/rolmedo/axo121/"

source "$VENV_PATH/bin/activate"
module load cuda/12.1

export HF_HOME=/tmp

export WANDB_PROJECT=$1

shift

$@