import transformers
import datasets
import torch
import sklearn
import numpy as np
import pandas as pd
import os
import gc
import pickle

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from accelerate import init_empty_weights
from dataclasses import dataclass, field
from prompter import Prompter

#peft_config = PeftConfig(peft_type="lora")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
from pynvml import *

from multiprocessing import Process
from lora_worker import run_lora_worker

def pop_peft(model):
    print('removing peft from model...')
    with init_empty_weights():
        lora_state, model_state = {}, {}
        for k, v in model.state_dict().items():
            if 'weight_format' in k or 'SCB' in k:
                pass
            elif 'lora' in k:
                lora_state[k] = v
            else:
                model_state[k.split('base_model.model.', 1)[1]] = v
        #config = LlamaConfig.from_pretrained(model_id, load_in_8bit=True, trust_remote_code=True)
        del model
        model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=False)
        model.load_state_dict(model_state)
        model = prepare_model_for_int8_training(model)

    del model_state
    #del lora_state
    torch.cuda.empty_cache()
    return model, lora_state

def push_peft(model, lora_state, lora_config):

    print('adding peft to model...')
    model = get_peft_model(model, lora_config)
    model_state_dict = model.state_dict()

    for k, v in lora_state.items():
        model_state_dict[k] = v

    model.load_state_dict(model_state_dict)
    return model

def merge_peft(model, lora_state, lora_config):

    print('merging peft...')

    model = get_peft_model(model, lora_config)
    model_state_dict = model.state_dict()

    for k, v in lora_state.items():
        model_state_dict[k] = v

    model.merge_and_unload()

    model, _ = pop_peft(model)

    return model

def merge_lora_adapters(lora_config, saved_lora_adapters):
    gc.collect()
    torch.cuda.empty_cache()
    model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto', load_in_8bit=False)

    for adapter in saved_lora_adapters:
        for k, v in adapter.items():
            adapter[k] = v.cuda()
        model = merge_peft(model, adapter, lora_config)
    return model

if __name__ == '__main__':
    for i in range(8):
        model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto', load_in_8bit=False)
