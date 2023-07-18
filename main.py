import transformers
import datasets
import torch
import sklearn
import numpy as np
import pandas as pd
import os
import gc
import pickle 
import subprocess

from pynvml import *

from transformers import (
    AutoModel,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaConfig,
    pipeline,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoConfig,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    )

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from accelerate import init_empty_weights
from dataclasses import dataclass, field
from prompter import Prompter
from multiprocessing import Process
from lora_worker import run_lora_worker

#################### MAKE EMBEDDINGS ####################

model_id = "elinas/llama-7b-hf-transformers-4.29"
dataset_id = "Fredithefish/ShareGPT-unfiltered-alpaca-lora-format"
ds = load_dataset(dataset_id, split="train")


prompter = Prompter('alpaca')
cutoff_len=256
train_on_inputs=True
add_eos_token=False
n_clusters=8

tokenizer = LlamaTokenizer.from_pretrained(model_id)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference

def to_wizardlm_prompt(ex):
    instruction = ex['instruction']
    response = ex['output']
    prompt = f"{instruction}\n\n### Response: {response}"
    return {
        "text": prompt
    }

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point['instruction'],
        data_point['input'],
        data_point['output']
    )
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

prompt_ds = ds.shuffle().map(generate_and_tokenize_prompt, num_proc=8)

if not os.path.exists('embed.npy'):

    embed_batch_size = 512
    # make embed model
    e_ds = ds.map(to_wizardlm_prompt)
    embed = SentenceTransformer('all-MiniLM-L6-v2')
    
    embed_ds = []
    l = len(e_ds)

    for i in tqdm( range(0, len(e_ds), embed_batch_size)):
        if i + embed_batch_size < l:
            _input = e_ds[i:i+embed_batch_size]['text']
            output = embed.encode(_input)
            for latent in output:
                embed_ds.append(latent)
        else:
            _input = e_ds[i:l-1]['text']
            output = embed.encode(_input)
    
            for latent in output:
                embed_ds.append(latent)
    
    embed_ds = np.array(embed_ds)

    np.save('embed', embed_ds) # cache the embedding ds
else:
    embed_ds = np.load('embed.npy')

#################### CLUSTER EMBEDDINGS ####################

if not os.path.exists('cluster_ds'):
    from sklearn.cluster import KMeans

    n_clusters = 8
    max_iter = 5000
    cl  = KMeans(n_clusters=n_clusters, max_iter=max_iter).fit(embed_ds)
    ds_clusters = [[] for i in range(8)]
    for i, label in tqdm(enumerate(cl.labels_)):
        ds_clusters[label].append(prompt_ds[i])
    
    #################### PREPARE DATASETS ####################
    
    seed = 42
    
    ds_list = list(map(lambda x: Dataset.from_pandas(pd.DataFrame(data=x)), ds_clusters))
    ds_dict = DatasetDict({str(idx): key for idx, key in enumerate(ds_list)})
    ds_dict = ds_dict.shuffle(seed=seed)

    ds_dict.save_to_disk('cluster_ds')
else:
    ds_dict = load_from_disk('cluster_ds')


#################### TRAIN ENSEMBLE MODEL ####################
# train n lora adapters on cluster ds

if __name__ == "__main__":
    base_model = "elinas/llama-7b-hf-transformers-4.29"


    from pynvml import *
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')

    outs = []
    for i in range(7, 8):
        out = subprocess.check_output(
            ['python3', 'lora_worker.py', '--dataset_dir', 'cluster_ds', '--cluster_idx', str(i)]
        )

        outs.append(out)
    

    with open('command_outputs.pickle', 'wb') as handle:
        pickle.dump(outs, handle)
    
    print('training finished')


    
