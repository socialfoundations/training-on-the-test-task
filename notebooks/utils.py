import os
import json
from tqdm import tqdm

import numpy as np
import seaborn as sns
palette = sns.color_palette("colorblind")

cn_families = {'qwen', 'yi', 'aquila', 'baichuan', 'internlm', 'internlm2', 'skywork', 'ziya2', 'map'}

model2family = {
    'stablelm-3b-4e1t': 'stablelm',
    'qwen-1.5-4b': 'qwen',
    'falcon-7b': 'falcon',
    'falcon-11b': 'falcon',
    'llama-2-7b': 'llama2',
    'falcon-rw-7b': 'falcon',
    'redpajama-7b': 'redpajama',
    'qwen-1.5-0.5b': 'qwen',
    'pythia-1b': 'pythia',
    'stablelm-base-alpha-3b-v2': 'stablelm',
    'pythia-410m': 'pythia',
    'pythia-1.4b': 'pythia',
    'qwen-1.5-14b': 'qwen',
    'ziya2-13b-base': 'ziya2',
    'falcon-rw-1b': 'falcon',
    'qwen-1.5-1.8b': 'qwen',
    'llama-2-13b': 'llama',
    'llama-2-70b': 'llama',
    'redpajama-3b': 'redpajama',
    'qwen-1.5-7b': 'qwen',
    'pythia-12b': 'pythia',
    'gpt-j-6b': 'gptj',
    'pythia-6.9b': 'pythia',
    'stablelm-base-alpha-7b-v2': 'stablelm',
    'pythia-160m': 'pythia',
    'yi-6b': 'yi',
    'pythia-2.8b': 'pythia',
    'pythia-70m': 'pythia',
    'olmo-1b': 'olmo',
    'olmo-1.7-7b': 'olmo',
    'olmo-7b': 'olmo',
    'gemma-2b': 'gemma',
    'gemma-7b': 'gemma',
    'llama-30b': 'llama',
    'llama-7b': 'llama',
    'llama-13b': 'llama',
    'llama-65b': 'llama',
    'baichuan2-13b': 'baichuan',
    'baichuan2-7b': 'baichuan',
    'baichuan-13b': 'baichuan',
    'baichuan-7b': 'baichuan',
    'internlm-20b': 'internlm',
    'internlm-7b': 'internlm',
    'internlm2-base-7b': 'internlm2',
    'internlm2-base-20b': 'internlm2',
    'openllama-13b': 'openllama',
    'openllama-3b': 'openllama',
    'openllama-3b-v2': 'openllama',
    'openllama-7b': 'openllama',
    'openllama-7b-v2': 'openllama',
    'stablelm-2-1.6b': 'stablelm2',
    'stablelm-3b-4e1t': 'stablelm',
    'stablelm-base-alpha-3b-v2': 'stablelm',
    'stablelm-base-alpha-7b': 'stablelm',
    'stablelm-base-alpha-7b-v2': 'stablelm',
    'llama-3-8b': 'llama3',
    'stablelm-2-12b': 'stablelm2',
    'skywork-13b': 'skywork',
    'map-neo-7b': 'map',
}

model2compute = {
    'qwen-1.5-4b': {'N': 4, 'D': 2400},
    'falcon-7b': {'N': 7, 'D': 1500},
    'llama-2-7b': {'N': 7, 'D': 2000},
    'redpajama-7b': {'N': 7, 'D': 1000},
    'qwen-1.5-0.5b': {'N': 0.5, 'D': 2400},
    'pythia-1b': {'N': 1, 'D': 300},
    'pythia-410m': {'N': 0.41, 'D': 300},
    'pythia-1.4b': {'N': 1.4, 'D': 300},
    'qwen-1.5-14b': {'N': 14, 'D': 4000},
    'qwen-1.5-1.8b': {'N': 1.8, 'D': 2400},
    'llama-2-13b': {'N': 13, 'D': 2000},
    'llama-2-70b': {'N': 70, 'D': 2000},
    'redpajama-3b': {'N': 3, 'D': 800},
    'qwen-1.5-7b': {'N': 7, 'D': 4000},
    'pythia-12b': {'N': 12, 'D': 300},
    'gpt-j-6b': {'N': 6, 'D': 400},
    'pythia-6.9b': {'N': 6.9, 'D': 300},
    'pythia-160m': {'N': 0.16, 'D': 300},
    'yi-6b': {'N': 6, 'D': 3000},
    'pythia-2.8b': {'N': 2.8, 'D': 300},
    'pythia-70m': {'N': 0.07, 'D': 300},
    'olmo-1b': {'N': 1, 'D': 2000},
    'olmo-7b': {'N': 7, 'D': 2460},
    'olmo-1.7-7b': {'N': 7, 'D': 2050},
    'gemma-2b': {'N': 2, 'D': 2000},
    'gemma-7b': {'N': 7, 'D': 6000},
    'llama-30b': {'N': 32.5, 'D': 1400},
    'llama-7b': {'N': 7, 'D': 1000},
    'llama-13b': {'N': 13, 'D': 1000},
    'llama-65b': {'N': 65.2, 'D': 1400},
    'baichuan2-13b': {'N': 13, 'D': 2600},
    'baichuan2-7b': {'N': 7, 'D': 2600},
    'baichuan-13b': {'N': 13, 'D': 1400},
    'baichuan-7b': {'N': 7, 'D': 1200},
    'internlm-20b': {'N': 20, 'D': 2300},
    'internlm-7b': {'N': 7, 'D': 1000},
    'internlm2-base-7b': {'N': 7, 'D': 2600},
    'internlm2-base-20b': {'N': 20, 'D': 2600},
    'openllama-13b': {'N': 13, 'D': 1000},
    'openllama-3b': {'N': 3, 'D': 1000},
    'openllama-3b-v2': {'N': 3, 'D': 1000},
    'openllama-7b': {'N': 7, 'D': 1000},
    'openllama-7b-v2': {'N': 7, 'D': 1000},
    'stablelm-2-1.6b': {'N': 1.6, 'D': 2000},
    'stablelm-3b-4e1t': {'N': 2.8, 'D': 4000},
    'stablelm-base-alpha-3b-v2': {'N': 2.8, 'D': 1100},
    'stablelm-base-alpha-7b-v2': {'N': 7, 'D': 1100},
    'falcon-11b': {'N': 11, 'D': 5000},
    'llama-3-8b': {'N': 8, 'D': 15000},
    'stablelm-2-12b': {'N': 12.1, 'D': 2000},
    'ziya2-13b-base': {'N': 13, 'D': 2650},
    'skywork-13b': {'N': 13, 'D': 3200},
    'map-neo-7b': {'N': 7, 'D': 4500},
}

model2date = {
    'qwen-1.5-4b': '2401',
    'falcon-7b': '2304',
    'llama-2-7b': '2307',
    'redpajama-7b': '2305',
    'qwen-1.5-0.5b': '2401',
    'pythia-1b': '2302',
    'pythia-410m': '2302',
    'pythia-1.4b': '2302',
    'qwen-1.5-14b': '2401',
    'qwen-1.5-1.8b': '2401',
    'llama-2-13b': '2307',
    'llama-2-70b': '2307',
    'redpajama-3b': '2305',
    'qwen-1.5-7b': '2401',
    'pythia-12b': '2302',
    'gpt-j-6b': '2103',
    'pythia-6.9b': '2302',
    'pythia-160m': '2302',
    'yi-6b': '2311',
    'pythia-2.8b': '2302',
    'pythia-70m': '2302',
    'olmo-1b': '2401',
    'olmo-7b': '2401',
    'olmo-1.7-7b': '2404',
    'gemma-2b': '2402',
    'gemma-7b': '2402',
    'llama-30b': '2302',
    'llama-7b': '2302',
    'llama-13b': '2302',
    'llama-65b': '2302',
    'baichuan2-13b': '2309',
    'baichuan2-7b': '2309',
    'baichuan-13b': '2306',
    'baichuan-7b': '2306',
    'internlm-20b': '2309',
    'internlm-7b': '2307',
    'internlm2-base-7b': '2401',
    'internlm2-base-20b': '2401',
    'openllama-13b': '2306',
    'openllama-3b': '2306',
    'openllama-3b-v2': '2307',
    'openllama-7b': '2306',
    'openllama-7b-v2': '2307',
    'stablelm-2-1.6b': '2401',
    'stablelm-3b-4e1t': '2309',
    'stablelm-base-alpha-3b-v2': '2308',
    'stablelm-base-alpha-7b-v2': '2308',
    'falcon-11b': '2405',
    'llama-3-8b': '2404',
    'stablelm-2-12b': '2403',
    'ziya2-13b-base': '2311',
    'skywork-13b': '2310',
    'map-neo-7b': '2405',
}

assert set(model2date.keys()) == set(model2compute.keys()), "mismatch between model2date and model2compute"

def get_model2date(model):
    return model2date[model]

def compute_compute(model):
    return model2compute[model]['N'] * model2compute[model]['D'] * 1e18 * 6

def can_compute(model):
    return model in model2compute

def get_brier_metrics(x, key=None):
    def f(x):
        results = {'acc': x['acc,none']}
        if 'brier_score,none' in x:
            results['brier'] = x['brier_score,none']
        return results
    
    if key is not None:
        return f(x[key])
    
    # else it is some dict
    x = list(x.values())
    results = [f(x_) for x_ in x]
    keys = results[0].keys()
    return {key: np.mean([r[key] for r in results]) for key in keys}

benchmarks = {
    'gsm8k': lambda x: x['gsm8k']['exact_match,flexible-extract'],

    'arc_challenge': lambda x: get_brier_metrics(x, 'arc_challenge'),
    'winogrande': lambda x: get_brier_metrics(x, 'winogrande'),
    'truthfulqa_mc2': lambda x: get_brier_metrics(x, 'truthfulqa_mc2'),
    'hellaswag': lambda x: get_brier_metrics(x, 'hellaswag'),

    'arc_challenge_mc': lambda x: get_brier_metrics(x, 'arc_challenge_mc'),
    'winogrande_mc': lambda x: get_brier_metrics(x, 'winogrande_mc'),
    'truthfulqa_mc2_mc': lambda x: get_brier_metrics(x, 'truthfulqa_mc2_mc'),
    'hellaswag_mc': lambda x: get_brier_metrics(x, 'hellaswag_mc'),

    'mmlu': lambda x: get_brier_metrics(x),
    'mmlu_cloze': lambda x: get_brier_metrics(x),

    'bbh_fewshot_mc': lambda x: get_brier_metrics(x, 'bbh_fewshot_mc'),
    'bbh_fewshot_cloze': lambda x: get_brier_metrics(x, 'bbh_fewshot_cloze'),

    'leaderboard_mmlu_pro': lambda x: x['leaderboard_mmlu_pro']['acc,none'],
    'leaderboard_mmlu_pro_four': lambda x: x['leaderboard_mmlu_pro_four']['acc,none'],
    'leaderboard_musr': lambda x: x['leaderboard_musr']['acc_norm,none'],
    'leaderboard_bbh': lambda x: x['leaderboard_bbh']['acc_norm,none'],
    'leaderboard_math_hard': lambda x: x['leaderboard_math_hard']['exact_match,none'],
    'leaderboard_gpqa': lambda x: x['leaderboard_gpqa']['acc_norm,none'],
}

def iter_files(base_dir, verbose=False, v2=False):
    for file in tqdm(os.listdir(base_dir)):
        model = '-'.join(file.split('-')[:-1])

        if not can_compute(model):
            if verbose and not model.startswith('pretrained____'):
                print(f"Skipping {model}")
            continue

        benchmark = file.split('-')[-1].split('.')[0]
        if benchmark not in benchmarks:
            continue

        file_dir = base_dir + file
        if v2: # new lm_eval format, find the actual .json file
            files = os.listdir(file_dir)[0]
            file_dir = file_dir + '/' + files
            files = os.listdir(file_dir)

        yield file_dir, model, benchmark

def read_benchmark_file(file, benchmarks, benchmark):
    with open(file) as f:
            data = json.load(f)
    
    data = benchmarks[benchmark](data['results'])
    if type(data) != dict:
        data = {'': data}

    return data

def load_benchmark_results(base_dir, benchmarks=benchmarks, verbose=False, v2=False):
    results = {}
    for file, model, benchmark in iter_files(base_dir, verbose=verbose, v2=v2):
        data = read_benchmark_file(file, benchmarks, benchmark)
        for bench, val in data.items():
            benchk = f"{benchmark}-{bench}" if bench else benchmark
            if benchk not in results:
                results[benchk] = {}
            results[benchk][model] = val

    return results

def load_steps(outer_dir, key):
    results = {}
    for model in tqdm(os.listdir(outer_dir)):
        if model not in model2compute:
            continue
        
        for step in os.listdir(outer_dir + model):
            if step.endswith('.json'):
                continue
            
            int_step = int(step.split('_')[-1])
            if not os.path.isdir(outer_dir + model + '/' + step):
                continue

            for file in os.listdir(outer_dir + model + '/' + step):
                benchmark = file.split('.')[0]
                
                if benchmark != key:
                    print(f"Skipping {benchmark}")
                    continue
                
                try:
                    with open(outer_dir + model + '/' + step + '/' + file) as f:
                        data = json.load(f)
                except Exception as e:
                    print(e)
                    continue

                data = benchmarks[benchmark](data['results'])

                if int_step not in results:
                    results[int_step] = {}
                
                results[int_step][model] = data

    return results

def cn_rule(model):
    return palette[1] if model2family[model] in cn_families else palette[0]

def color_rule(model, use_date=True, threshold ='2311'):
    if use_date:
        return palette[1] if get_model2date(model) >= threshold else palette[0]
    return cn_rule(model)

def process_pre_post_adjustment(pre, post, use_max=True, verbose=False):
    if verbose:
        print('---------------')
        pre_not_post = set(pre.keys()) - set(post.keys())
        if pre_not_post:
            print("Models in pre not in post", pre_not_post)
        post_not_pre = set(post.keys()) - set(pre.keys())
        if post_not_pre:
            print("Models in post not in pre", pre_not_post)

    # ensure same models in pre and post
    models = set(pre.keys()) & set(post.keys())
    pre = {m: pre[m] for m in models}
    post = {m: post[m] for m in models}

    if use_max:
        post = {m: max(pre[m], post[m]) for m in pre}

    if verbose:
        print(f"Total models: {len(pre)}")

    return pre, post