"""
Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
from the Accelerate example zoo https://huggingface.co/docs/accelerate/usage_guides/training_zoo
"""

import os
import shutil

from dataclasses import field, dataclass
from typing import Optional

import torch
import transformers
from dataclasses import dataclass
from transformers import TrainerCallback
from jobs_lmeval_leaderboard import launch_lmeval_job
from get_reqs import get_n_gpus, get_job_memory, get_model_name
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

IGNORE_INDEX = -100


def load_model_tokenizer(model_name_or_path, tokenizer_name=None):
    assert model_name_or_path, "You must pass a model_name_or_path"

    # Load the tokenizer
    tokenizer_kwargs = {
        'pretrained_model_name_or_path': tokenizer_name if tokenizer_name else model_name_or_path,
        'trust_remote_code': True,
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(**tokenizer_kwargs)

    model_kwargs = {
        'pretrained_model_name_or_path': model_name_or_path,
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
    }

    print('Device count', torch.cuda.device_count())
    if torch.cuda.device_count() == 1:
        model_kwargs['device_map'] = 'auto'

    def class_to_use(model_name):
        if 'falcon' in model_name.lower():
            return transformers.FalconForCausalLM
        return transformers.AutoModelForCausalLM

    # try to load the model with flash_attention_2
    try:
        if torch.cuda.get_device_capability(0)[0] < 8:
            raise ValueError("Flash attention only works for CUDA compatibility >= 8")
        
        model = class_to_use(model_name_or_path).from_pretrained(
            attn_implementation="flash_attention_2",
            **model_kwargs
        )
        print("Model loaded with flash attention")
        tokenizer.padding_side = 'left'
    except:
        model = class_to_use(model_name_or_path).from_pretrained(
            **model_kwargs
        )
        print("Model loaded without flash attention")

    if 'qwen' in model_name_or_path.lower():
        tokenizer.padding_side = 'left'

    # print the dtype of the model
    print(f"Model dtype: {model.dtype}")

    if tokenizer.pad_token is None:
        print("Setting pad token to EOS")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # We resize the embeddings only when necessary to avoid index errors.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size and 'baichuan2' not in model_name_or_path:
        # baichuan leads to error, see https://github.com/baichuan-inc/Baichuan2/issues/49
        print("Resizing the embeddings to match the tokenizer size.")
        model.resize_token_embeddings(len(tokenizer))
        print("Resized the embeddings.")

    return model, tokenizer

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

class SaveAndEvalCallback(TrainerCallback):
    def __init__(self, eval_type, model_output_dir, eval_save_dir):
        self.output_dir = model_output_dir
        self.eval_save_dir = eval_save_dir
        self.steps_logged = set()

        if 'mmlu' in eval_type:  # total is 4,5k steps
            self.steps_of_interest = [10, 30, 100, 250, 500]
            self.multiplier_of_interest = 1000
            self.tasks = {
                'mmlu': {'args': '--num_fewshot 5'},
            }
        elif 'gsm8k' in eval_type:  # total is 25k
            self.steps_of_interest = [10, 30, 100, 250, 500, 1000, 2500]
            self.multiplier_of_interest = 2500
            self.tasks = {
                'gsm8k': {'args': '--num_fewshot 5'},
            }
        elif eval_type.startswith('ours_'):
            tasks = [eval_type[5:]]
            self.steps_of_interest = [5, 10, 20, 30, 40, 60, 80, 160, 320, 500]
            self.multiplier_of_interest = 1000
            self.tasks = {
                task: {} for task in tasks
            }
        else:
            raise ValueError("Evaluation type not recognized")
        
    def set_trainer(self, trainer, tokenizer):
        self.trainer = trainer
        self.tokenizer = tokenizer

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step

        def check_interest(step):
            if step in self.steps_logged:
                return False
            if step in self.steps_of_interest:
                return True
            if step % self.multiplier_of_interest == 0 and step > 0:
                return True
            return False

        if check_interest(step):
            print(f"Step {step} is of interest, saving model and launching eval jobs.")
            self.steps_logged.add(step)
            self.save_model(step, model)
            if state.is_world_process_zero:
                self.launch_eval_job(step)

    def get_save_dir(self, step):
        return f"{self.output_dir}/step_{step}"
    
    def get_eval_save_dir(self, step):
        return f"{self.eval_save_dir}/step_{step}"

    def save_model(self, step, model):
        output_dir = self.get_save_dir(step)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model at step {step}")

        if is_deepspeed_zero3_enabled():
            save_zero3(self.trainer, self.tokenizer, output_dir)
        else:
            self.tokenizer.save_pretrained(output_dir)
            model.save_pretrained(output_dir, save_config=True)

    def launch_eval_job(self, step):
        output_dir = self.get_save_dir(step)
        eval_save_dir = self.get_eval_save_dir(step)
        os.makedirs(eval_save_dir, exist_ok=True)
        print(f"Launching eval jobs for step {step}")
        for task, args in self.tasks.items():
            save_dir = f"{eval_save_dir}/{task}.json"
            additional_kwargs = args['args'] if 'args' in args else ''
            launch_lmeval_job(
                model_dir=output_dir,
                tasks=[task],
                save_file=save_dir,
                additional_kwargs=additional_kwargs,
                JOB_MEMORY="64GB",
                JOB_CPUS=4,
                JOB_GPUS=get_n_gpus(output_dir),
                GPU_MEM=get_job_memory(output_dir),
                JOB_BID=101,
                rm_model=True,
            )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    block_size: Optional[int] = field(
        default=2048,
        metadata={"help": "Block size for the model."},
    )
    lr_scheduler_min_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum learning rate for cosine annealing."},
    )
    saves_per_epoch: Optional[int] = field(
        default=None,
        metadata={"help": "Number of evaluations per epoch."},
    )
    save_and_eval_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save the model for evaluation."},
    )
    save_and_eval_model_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save the model for evaluation."},
    )
    checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to save checkpoints."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_task: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    eval_size: Optional[int] = field(
        default=3000,
        metadata={"help": "Number of examples to evaluate on."},
    )
    eval_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save the eval results."},
    )
    
def save_zero3(trainer, tokenizer, output_dir):
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=trainer.accelerator.is_main_process,
        save_function=trainer.accelerator.save,
        state_dict=trainer.accelerator.get_state_dict(trainer.model_wrapped),
        safe_serialization=False,
    )

    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    from data_utils import load_instructions
    from transformers import HfArgumentParser, Trainer, TrainingArguments

    # https://huggingface.co/docs/transformers/v4.41.2/en/main_classes/trainer#transformers.TrainingArguments

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # add lr_scheduler_min_lr to training_args
    if model_args.lr_scheduler_min_lr and training_args.lr_scheduler_type == 'cosine_with_min_lr':
        if is_deepspeed_zero3_enabled():
            training_args.lr_scheduler_kwargs = {'cos_min_ratio': model_args.lr_scheduler_min_lr}
        else:
            training_args.lr_scheduler_kwargs = {'min_lr' : model_args.lr_scheduler_min_lr * training_args.learning_rate}

    # load model and tokenizer
    model, tokenizer = load_model_tokenizer(model_args.model_name_or_path)

    # load training data
    train_dataset = load_instructions(
        data_args.train_task,
        tokenizer,
        max_length=model_args.block_size,
    )

    # set eval steps and save steps
    if model_args.saves_per_epoch is not None:
        global_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        global_batch_size *= torch.cuda.device_count()
        steps_per_epoch = int(len(train_dataset) / global_batch_size)
        training_args.eval_steps = int(steps_per_epoch / model_args.saves_per_epoch)
        training_args.save_strategy = 'steps'
        training_args.save_steps = training_args.eval_steps

    if not model_args.checkpoint and not is_deepspeed_zero3_enabled():
        training_args.save_only_model = True

    # callbacks --- early stopping and evaluation
    callbacks = []

    use_save_and_eval = model_args.save_and_eval_save_dir and model_args.save_and_eval_model_output_dir
    if use_save_and_eval:
        save_and_eval_callback = SaveAndEvalCallback(
            eval_type=data_args.train_task,
            model_output_dir=model_args.save_and_eval_model_output_dir,
            eval_save_dir=model_args.save_and_eval_save_dir,
        )
        callbacks.append(save_and_eval_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Training
    if use_save_and_eval:
        save_and_eval_callback.set_trainer(trainer, tokenizer)
    resume_from_checkpoint = any('checkpoint' in file for file in os.listdir(training_args.output_dir))
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # end of training
    if is_deepspeed_zero3_enabled():
        trainer.accelerator.wait_for_everyone()
    
    # print('Saving model...')
    if is_deepspeed_zero3_enabled():
        save_zero3(trainer, tokenizer, training_args.output_dir)
    else:
        trainer.save_model(output_dir=training_args.output_dir)

    if not trainer.accelerator.is_main_process:
        exit(0)

    # remove all folders where the name contains 'checkpoint'
    for file in os.listdir(training_args.output_dir):
        if 'checkpoint' in file:
            file_path = os.path.join(training_args.output_dir, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
    
    # launch eval script
    if data_args.eval_save_dir is None:
        exit(1)

    if 'mmlu' in data_args.train_task:
        tasks = {
            'mmlu': {'args': '--num_fewshot 5'},
            'arc_challenge_mc': {'args': '--num_fewshot 25'},
            'truthfulqa_mc2_mc': {'args': '--num_fewshot 6'},
            'hellaswag_mc': {'args': '--num_fewshot 10'},
            'winogrande_mc': {'args': '--num_fewshot 5'},
        }
    elif 'gsm8k' in data_args.train_task:
        tasks = {
            'gsm8k': {'args': '--num_fewshot 5'},
            'minerva_math': {'args': ''},
        }
    else:
        print("No eval task specified")
        exit(1)

    model_name = get_model_name(training_args.output_dir)
    for task in tasks:
        launch_lmeval_job(
            model_dir=training_args.output_dir,
            tasks=[task],
            save_file=f"{data_args.eval_save_dir}/{model_name}-{task}.json",
            additional_kwargs=tasks[task]['args'],
            JOB_MEMORY="64GB",
            JOB_CPUS=4,
            JOB_GPUS=get_n_gpus(model_name),
            GPU_MEM=get_job_memory(model_name),
            JOB_BID=101,
        )
