# These functions are used to load data from lm_eval tasks and convert them into tokenized datasets
import datasets

def mmluaux_row_to_qa(row):
    input_ = f'{row["question"].strip()}\n'
    for i, choice in enumerate(row['choices']):
        input_ += f'{chr(65 + i)}. {choice.strip()}\n'
    input_ += f"Answer:"
    target = ' ' + chr(65 + row['answer'])
    return input_, target

def gsm8k_rows_to_qa(row, header=''):
    prompt = header[:]
    prompt += f"Question: {row['question']}\nAnswer:"
    return prompt, row['answer']

def tokenize_dataset(dataset, tokenizer, max_length, add_eos=True, pad_to_max_length=True, remove_too_long=False):
    def process_input_ids(array_input_ids, lens_input_tok):
        for input_ids, l in zip(array_input_ids, lens_input_tok):
            if add_eos:
                input_ids += [tokenizer.eos_token_id]

            if remove_too_long and len(input_ids) > max_length:
                continue
            
            labels = [-100] * l + input_ids[l:]

            # at most max_length
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]

            if pad_to_max_length:
                if tokenizer.padding_side == 'right':
                    input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
                    labels = labels + [-100] * (max_length - len(labels))
                else:
                    input_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids)) + input_ids
                    labels = [-100] * (max_length - len(labels)) + labels
                assert len(input_ids) == max_length
            
            assert len(input_ids) == len(labels)

            yield input_ids, labels

    def map_tokenize(examples):
        input_answer = [inp + ans for inp, ans in zip(examples['inputs'], examples['targets'])]
        input_ids = tokenizer(input_answer).input_ids
        len_input_tok = [len(tok) for tok in tokenizer(examples['inputs']).input_ids]
        input_ids_labels = tuple(map(list, zip(*process_input_ids(input_ids, len_input_tok))))
        return {'input_ids': input_ids_labels[0], 'labels': input_ids_labels[1]}        
    
    return dataset.map(map_tokenize, batched=True, remove_columns=dataset.column_names)

def process_math(dataset, max_chars=1800):
    def map(row):   
        input_, target = gsm8k_rows_to_qa(row)
        return {'inputs': input_, 'targets': target}
    dataset = dataset.map(map)

    # filter those <= max_chars
    n_chars = lambda x: len(x['inputs']) + len(x['targets'])
    dataset = dataset.filter(lambda x: n_chars(x) <= max_chars)
    return dataset

def load_orca_math(max_chars):
    dataset = datasets.load_dataset('microsoft/orca-math-word-problems-200k')['train']
    return process_math(dataset, max_chars)

def load_metamathqa(max_chars):
    dataset = 'meta-math/MetaMathQA'
    data = datasets.load_dataset(dataset)['train']
    data = data.rename_column('query', 'question')
    data = data.rename_column('response', 'answer')
    data = data.remove_columns(['type', 'original_question'])
    return process_math(data, max_chars)

def load_gsm8kaux(tokenizer, max_length, max_chars=1700):
    dataset1 = load_orca_math(max_chars)
    dataset2 = load_metamathqa(max_chars)
    dataset = datasets.concatenate_datasets([dataset1, dataset2]).shuffle(seed=42)
    return tokenize_dataset(dataset, tokenizer, max_length)

def load_mmluaux(tokenizer, max_length):
    dataset = datasets.load_dataset('cais/mmlu', 'all')['auxiliary_train']
    def mmluaux_to_qa_prompt(row):
        input_, target = mmluaux_row_to_qa(row)
        return {'inputs': input_, 'targets': target}
    dataset = dataset.map(mmluaux_to_qa_prompt)
    return tokenize_dataset(dataset, tokenizer, max_length)

def load_instructions(name, tokenizer, max_length, **kwargs):
    f = {
        'mmluaux': load_mmluaux,
        'gsm8kaux': load_gsm8kaux,
    }.get(name)
    return f(tokenizer, max_length, **kwargs)
