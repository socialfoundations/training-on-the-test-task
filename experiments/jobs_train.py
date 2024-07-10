#!/usr/bin/env python3

from pathlib import Path
import htcondor

JOB_BID = 55

accelerate_template = """compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: zero3_config/zero3.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: {num_processes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

def launch_experiment_job(
        CLUSTER_LOGS_SAVE_DIR,
        run_name,
        train_task,
        model_path,
        output_dir=None,
        eval_save_dir=None,
        num_train_epochs=None,
        max_train_steps=None,
        block_size=4096,
        gradient_accumulation_steps=1,
        lr=2e-6,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_and_eval_save_dir=None,
        save_and_eval_model_output_dir=None,
        JOB_GPUS=1,
        GPU_MEM=None,
        gradient_checkpointing=False,
        checkpoint=False,
        saves_per_epoch=4,
        JOB_MEMORY=80,
        JOB_CPUS=8,
        i=0,
        max_price=None,
        N_NODES=None,
        **kwargs,
):
    print('\n\n')
    print(run_name)
        
    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    base_acc_config_dir = 'configs/tmp/'
    Path(base_acc_config_dir).mkdir(parents=True, exist_ok=True)

    executable = 'empty_bash.sh'
    if N_NODES is not None:
        assert N_NODES == 2
        header = "accelerate launch --config_file zero3_config/two_nodes.yaml train.py "
        gradient_accumulation_steps = int(gradient_accumulation_steps / JOB_GPUS / N_NODES)
    elif JOB_GPUS > 1:
        header = f"accelerate launch --main_process_port {29500+1+i} --config_file {base_acc_config_dir}accelerate_config_{run_name}.yaml train.py "
        accelerate_config = accelerate_template.format(
            num_processes=JOB_GPUS,
        )
        with open(f'{base_acc_config_dir}accelerate_config_{run_name}.yaml', 'w') as f:
            f.write(accelerate_config)
        gradient_accumulation_steps = int(gradient_accumulation_steps / JOB_GPUS)
    else:
        header = "python train.py"
    header = 'ttt ' + header

    train_command = (
        f"{header} "
        f"--model_name_or_path {model_path} "
        f"--train_task {train_task} "
        f"--block_size {block_size} "
        f"--run_name {run_name} "
        f"--gradient_accumulation_steps {gradient_accumulation_steps} "
        f"--per_device_train_batch_size {per_device_train_batch_size} "
        f"--per_device_eval_batch_size {per_device_eval_batch_size} "
        f"--learning_rate {lr} "
        f"--adam_beta1 0.9 "
        f"--adam_beta2 0.95 "
        f"--adam_epsilon 1e-8 "
        f"--weight_decay 0.1 "
        f"--max_grad_norm 1.0 "
        f"--bf16 "
        f"--optim adamw_torch_fused "
        f"--report_to wandb "
        f"--lr_scheduler_type cosine_with_min_lr "
        f"--lr_scheduler_min_lr 0.1 "
        f"--warmup_ratio 0.01 "
        f"--save_total_limit 1 "
        f"--logging_steps 1 "
    )
    if output_dir is None:
        output_dir = "/tmp/"  # removed once the job finishes
    if output_dir is not None:
        train_command += f"--output_dir {output_dir} "
        train_command += "--overwrite_output_dir "
    if num_train_epochs is not None:
        train_command += f"--num_train_epochs {num_train_epochs} "
    if max_train_steps is not None:
        train_command += f"--max_train_steps {max_train_steps} "
    if gradient_checkpointing:
        train_command += "--gradient_checkpointing "
        train_command += "--save_strategy no "
    if checkpoint:
        train_command += "--checkpoint "
        train_command += "--save_strategy steps "
        train_command += f"--saves_per_epoch {saves_per_epoch} "
        train_command += f"--evaluation_strategy steps "
    else:
        train_command += "--save_strategy no "
        train_command += "--evaluation_strategy no "
    if eval_save_dir is not None:
        train_command += f"--eval_save_dir {eval_save_dir} "
    if save_and_eval_save_dir is not None:
        train_command += f"--save_and_eval_save_dir {save_and_eval_save_dir} "
    if save_and_eval_model_output_dir is not None:
        train_command += f"--save_and_eval_model_output_dir {save_and_eval_model_output_dir} "

    all_commands = train_command

    if N_NODES is not None and N_NODES > 1:
        # these jobs are submitted manually, not through condor
        print(all_commands)
        return

    job_settings = {
        "executable": f"{executable}",
        "arguments": f"{all_commands}",
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_gpus": f"{JOB_GPUS}",
        "request_cpus": f"{max(JOB_CPUS*JOB_GPUS, 32)}",  # how many CPU cores we want
        "request_memory": f"{JOB_MEMORY*JOB_GPUS}GB",  # how much memory we want
        "request_disk": f"{JOB_MEMORY*JOB_GPUS}GB",
        "jobprio": f"{JOB_BID - 1000}",
    }

    if max_price is not None:
        job_settings["+MaxRunningPrice"] = max_price
        job_settings["+RunningPriceExceededAction"] = htcondor.classad.quote('restart')

    cuda_cap = 9 if JOB_GPUS > 1 else 8  # only H100s for multi-gpu
    if GPU_MEM is not None:
        req = f"(TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}) && (TARGET.CUDACapability >= {cuda_cap}.0)"

    if req:
        job_settings["requirements"] = req

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")

if __name__ == '__main__':
    from data_utils import models

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_task', type=str, required=True)
    parser.add_argument('--save_intermediate', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--onlyold', action='store_true')
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--base_model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--eval_save_dir', type=str, default=None)
    parser.add_argument('--save_and_eval_dir', type=str, default=None)

    args = parser.parse_args()
    train_task = args.train_task
    output_dir = args.output_dir

    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.eval_save_dir is not None:
        eval_save_dir = f'{args.eval_save_dir}{train_task}/'
        Path(eval_save_dir).mkdir(parents=True, exist_ok=True)

    batch_size = 64  # global batch size
    block_size = 600  # 95% of examples have length <= 600 for both mmluaux and gsm8kaux
    batch_multiplier = 4  # use larger batch sizes, since smaller context window

    if args.onlyold:
        models = {k: v for k, v in models.items() if not v['isnew']}

    if args.base_model_dir is not None:
        for model in models:
            target_dir = args.base_model_dir + model
            if not Path(target_dir).exists():
                target_dir = None
            models[model]['model_path'] = target_dir
        models = {k: v for k, v in models.items() if v['model_path'] is not None}

    for model_args in models.values():
        if 'JOB_GPUS' not in model_args:  # do not alter the batch size for multi-gpu models
            model_args['per_device_train_batch_size'] = int(batch_multiplier * model_args['per_device_train_batch_size'])

    if args.save_intermediate:
        save_and_base_eval_kargs = {
            'save_and_eval_save_dir': f'{args.save_and_eval_dir}{train_task}/evals/',
            'save_and_eval_model_output_dir': f'{args.save_and_eval_dir}{train_task}/models/',
        }
    else:
        save_and_base_eval_kargs = {}

    if 'save_and_eval_save_dir' in save_and_base_eval_kargs:
        Path(save_and_base_eval_kargs['save_and_eval_save_dir']).mkdir(parents=True, exist_ok=True)

    for model_name, model_args in models.items():
        model_output_dir = None if output_dir is None else f'{output_dir}{train_task}/{model_name}/'
        if model_output_dir is not None:
            Path(model_output_dir).mkdir(parents=True, exist_ok=True)

        # if the output dir exists, and is not empty, skip
        if model_output_dir is not None:
            if Path(model_output_dir).exists() and len(list(Path(model_output_dir).iterdir())) > 0 and not args.overwrite:
                continue

        JOB_GPUS = model_args['JOB_GPUS'] if 'JOB_GPUS' in model_args else 1
        grad_acc_steps = int(batch_size / model_args['per_device_train_batch_size'])
        model_name_ = f"foundation_{model_name}-{train_task}"

        my_save_and_eval_kargs = {k: v + model_name + '/' for k, v in save_and_base_eval_kargs.items()}
        if 'save_and_eval_save_dir' in my_save_and_eval_kargs:
            Path(my_save_and_eval_kargs['save_and_eval_save_dir']).mkdir(parents=True, exist_ok=True)

        launch_experiment_job(
            Path('/fast/rolmedo/logs/'),
            run_name=model_name_,
            train_task=train_task,
            output_dir=model_output_dir,
            num_train_epochs=args.num_train_epochs,
            gradient_accumulation_steps=grad_acc_steps,
            block_size=block_size,
            eval_save_dir=args.eval_save_dir,
            **my_save_and_eval_kargs,
            **model_args,
        )
