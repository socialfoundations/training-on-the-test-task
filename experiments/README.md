# Experiments

Code to reproduce the experiments in the paper.

### Create venv with all requirements

Change `VENV_PATH` in all bash `.sh` scripts to math your system. Then run:

```bash
./install_requirements.sh
```

### Experiment 0: evaluate benchmark scores of the base models

Change `BASE_MODEL_PATH` in to the path where the base models are located.
You may download the models from the provided data repository, or from HF directly (check `hf` in `model_utils.py`).

Change `SAVE_DIR` to match your system. For example `export SAVE_DIR='/fast/groups/sf/ttt/'`.

```bash
python jobs_eval.py --base_save_dir "${SAVE_DIR}/evaluations/base/"
```

### Experiment 1: fine-tune all models on mmluaux and gsm8kaux for 3 epochs

Then, run:

```bash
python jobs_train.py \
    --train_task mmluaux \
    --save_intermediate \
    --num_train_epochs 3 \
    --output_dir "${SAVE_DIR}models/e3/" \
    --eval_save_dir "${SAVE_DIR}evaluations/e3/" \
    --save_and_eval_dir "${SAVE_DIR}intermediate/"
```

Note that this saves intermediate checkpoints for the section on emergence.

For gsm8kaux, simply change `mmluaux` for `gsm8kaux`.

### Experiment 2: fine-tune old models for 1 and 2 epochs

```bash
python jobs_train.py \
    --train_task mmluaux \
    --num_train_epochs 1 \
    --output_dir "${SAVE_DIR}models/e1/" \
    --eval_save_dir "${SAVE_DIR}evaluations/e1/" \
    --onlyold

python jobs_train.py \
    --train_task mmluaux \
    --num_train_epochs 2 \
    --output_dir "${SAVE_DIR}models/e2/" \
    --eval_save_dir "${SAVE_DIR}evaluations/e2/" \
    --onlyold
```

Once the jobs for 1 epoch have finished, fine-tune those models for an additional 2 epochs:

```bash
python jobs_train.py \
    --train_task mmluaux \
    --num_train_epochs 2 \
    --output_dir "${SAVE_DIR}models/e1+2/" \
    --eval_save_dir "${SAVE_DIR}evaluations/e1+2/" \
    --base_model_dir "${SAVE_DIR}models/e1/mmluaux/"
```

For gsm8kaux, simply change `mmluaux` for `gsm8kaux`.
