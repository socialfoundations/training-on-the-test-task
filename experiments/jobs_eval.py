import os
from pathlib import Path

import htcondor

JOB_BID_SINGLE = 100
JOB_BID_MULT = 150

def launch_lmeval_job(
        model_dir,
        tasks,
        save_file,
        additional_kwargs,
        JOB_MEMORY,
        JOB_CPUS,
        JOB_GPUS=1,
        use_bf16=False,
        GPU_MEM=None,
        JOB_BID=JOB_BID_SINGLE,
        rm_model=False,
):
    # Name/prefix for cluster logs related to this job
    CLUSTER_LOGS_SAVE_DIR=Path('/fast/rolmedo/logs/')
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    pretrained = f"pretrained={model_dir}"
    if 'falcon' not in model_dir.lower():
        pretrained += ',trust_remote_code=True'

    if use_bf16:
        pretrained += ',dtype=bfloat16'

    if JOB_GPUS > 1:
        pretrained += ',parallelize=True'

    executable = 'lm_eval_rm.sh' if rm_model else 'lm_eval.sh'

    # Construct job description
    job_settings = {
        "executable": executable,
        "arguments": (
            f"{pretrained} "
            f"{','.join(tasks)} "
            f"{save_file} "
            f"{model_dir} "
            f"{additional_kwargs} "
        ),
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_cpus": f"{JOB_CPUS}",  # how many CPU cores we want
        "request_gpus": f"{JOB_GPUS}",
        "request_memory": JOB_MEMORY,  # how much memory we want
        "request_disk": JOB_MEMORY,
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "rdo@tue.mpg.de",  # otherwise one does not notice an you can miss clashes
        "notification": "error",
    }

    if GPU_MEM is not None:
        job_settings["requirements"] = f"(TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}) && (CUDACapability >= 8.0)"
    else:
        job_settings["requirements"] = "CUDACapability >= 8.0"

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")

if __name__ == "__main__":
    from model_utils import models, memory, get_n_gpus

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--base_save_dir', type=str, required=True)  # e.g., /fast/groups/sf/ttt/evaluations/base/
    parser.add_argument('--base_model_dir', type=str, default=None)  # e.g., /fast/groups/sf/ttt/models/mmluaux/e3/

    args = parser.parse_args()

    models = {k: v['model_path'] for k, v in models.items()}
    base_save_dir = args.base_save_dir

    task = args.task
    if task is not None:
        base_folder = args.base_model_dir
        assert base_folder is not None
        
        files = os.listdir(base_folder)
        models = {f: base_folder + f for f in files}
        base_save_dir = args.base_save_dir + task

    qa_benchmarks = {
        'arc_challenge': {'args': '--num_fewshot 25'},
        'truthfulqa_mc2': {},
        'hellaswag': {'args': '--num_fewshot 10'},
        'winogrande': {'args': '--num_fewshot 5'},
        'mmlu': {'args': '--num_fewshot 5'},
    }

    mc_benchmarks = {
        'arc_challenge_mc': {'args': '--num_fewshot 25'},
        'truthfulqa_mc2_mc': {'args': '--num_fewshot 6'},
        'hellaswag_mc': {'args': '--num_fewshot 10'},
        'winogrande_mc': {'args': '--num_fewshot 5'},
    }

    math_benchmarks = {
        'gsm8k': {'args': '--num_fewshot 5'},
    }

    if args.task is None:
        tasks = {**qa_benchmarks, **mc_benchmarks, **math_benchmarks}
    else:
        if 'mmlu' in args.task:
            tasks = {**qa_benchmarks, **mc_benchmarks}
        elif 'gsm8k' in args.task:
            tasks = math_benchmarks
        else:
            raise ValueError(f"Unknown task {args.task}")

    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)

    shared_args = ""
    for model, model_dir in models.items():
        for task_name, args in tasks.items():
            save_name = f"{model}-{task_name}"
            save_file = f"{base_save_dir}/{save_name}.json"

            # if save file exists, skip
            if os.path.exists(save_file):
                continue

            # if the model dir is empty, skip
            if len(os.listdir(model_dir)) == 0:
                continue

            additional_kwargs = args['args'] if 'args' in args else ''
            additional_kwargs += shared_args

            if 'JOB_GPUS' in args:
                del args['JOB_GPUS']

            if not os.path.exists(save_file):
                GPU_MEM = memory[model] if model in memory else None
                use_bf16 = GPU_MEM is not None and GPU_MEM > 39000
                print(f"Launching {save_file}")
                launch_lmeval_job(
                    model_dir=model_dir,
                    tasks=[args['task']] if 'task' in args else [task_name],
                    save_file=save_file,
                    additional_kwargs=additional_kwargs,
                    JOB_MEMORY="64GB",
                    JOB_CPUS="8",
                    JOB_GPUS=get_n_gpus(model),
                    GPU_MEM=GPU_MEM,
                    JOB_BID=JOB_BID_MULT if get_n_gpus(model) > 1 else JOB_BID_SINGLE,
                )
