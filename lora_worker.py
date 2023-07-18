import transformers
import argparse
import datasets
import torch
import sklearn
import numpy as np
import pandas as pd
import os
import gc
import pickle

from pynvml import *
from datasets import load_from_disk

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

model_id = "elinas/llama-7b-hf-transformers-4.29"
dataset_id = "WizardLM/WizardLM_evol_instruct_V2_196k"

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
        #model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto')
        #model.load_state_dict(model_state)
        #model = prepare_model_for_int8_training(model)

    #del model_state
    #torch.cuda.empty_cache()
    return lora_state

def run_lora_worker(dataset_dir: str, cluster_idx: int, base_model: str):
    """
    train a single LoRA adapter and save it as a pickle
    """

    cluster_idx = str(cluster_idx) 

    train_data = load_from_disk('cluster_ds')['0']

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map='auto'
    )
    model= prepare_model_for_int8_training(model)


    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj']
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model
    )

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    coll = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')


    grad_acc_steps = 8
    num_epochs = 3
    lora_alpha = 16
    lora_r = 8
    base_wandb_run_name = 'lora'

    model = get_peft_model(model, lora_config)
    learning_rate = 3e-4

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            warmup_steps=500,
            num_train_epochs=3,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adamw_torch",
            save_strategy="steps",
            output_dir='output',
            fp16=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ))

    trainer.train()
    lora_adapter = pop_peft(model)

    output_name = 'run_' + cluster_idx + '.pickle'

    with open(output_name, 'wb') as handle:
        pickle.dump(lora_adapter, handle)
    
    return


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir',
                        type=str,
                        help='directory of the cluster dataset')
    parser.add_argument('--cluster_idx',
                        type=str,
                        help='id of the cluster')
    parser.add_argument('--base_model',
                        type=str,
                        default="elinas/llama-7b-hf-transformers-4.29",
                        help='the seq2seq base model to use')
    args = parser.parse_args()
    run_lora_worker(args.dataset_dir, args.cluster_idx, args.base_model)
