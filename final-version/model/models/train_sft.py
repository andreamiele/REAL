# THIS FILE SHOULD BE RUN FROM THE MODEL DIRECTORY.

import pandas as pd
import json
import sys 
sys.path.append('model')
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import load_dataset
from datasets import Dataset
import transformers
from transformers import TrainingArguments
from datasets import concatenate_datasets
import argparse
import torch
from peft import LoraConfig, TaskType, get_peft_model
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import wandb
def formatting_prompts_func(example, tokenizer):
    data = []
    for i in range(len(example['question'])):
        instruction = example['question'][i]
        completion = example['answer'][i]
        prompt = f"""{tokenizer.bos_token} [INST] ### Task: You will be asked a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. Your task is to answer the question to the best of your abilities. You must complete the task step by step and give your final answer by completing the following json: {'{'} 'Answer': ... {'}'}.\
                    You must explain your reasoning process.  Ensure that your answer is detailed, accurate, and logical, demonstrating a deep understanding of the topic. Remember, clarity, coherence, and accuracy are key components of a successful response.\
                        ### {instruction}\n### Answer: {completion} {tokenizer.eos_token}"""
        data.append(prompt)
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='google/gemma-2b')
    parser.add_argument('--n-epochs', default=4, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--eval-steps', default=80, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=2e-4, type=float)
    parser.add_argument('--warmup-steps', default=0, type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=80, type=int)
    parser.add_argument('--logging-steps', default=80, type=int)
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--max-length', default=512, type=int)
    parser.add_argument('--use-peft', default='false')
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=32, type=int)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    return parser.parse_args()


def get_training_args(args):
        return SFTConfig(
            output_dir=args.output_dir,               
            overwrite_output_dir=False,                  
            num_train_epochs=args.n_epochs,                   
            per_device_train_batch_size=args.batch_size,         
            learning_rate=args.learning_rate,                      
            warmup_steps=args.warmup_steps,                           
            weight_decay=args.weight_decay,                         
            adam_epsilon=args.adam_epsilon,                         
            save_steps=args.save_steps,       
            logging_strategy='steps',                
            logging_steps=args.logging_steps,                      
            save_total_limit=1,                         
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )       

def main():
    print('start')
    args = parse_args()
    model_name = args.model_name
    
    train_data = load_dataset('json', data_files='datasets/M1_reformatted.json', split='train[:25000]')

    output_directory =f'{args.output_dir}/sft_{model_name.split("/")[-1]}'
    args.output_dir = output_directory.replace(' ', '_')
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', cache_dir='cache')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token=tokenizer.eos_token
    print('model loaded')
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.peft_config_r, lora_alpha=args.peft_config_lora_alpha, lora_dropout=args.peft_config_lora_dropout)
    training_args = get_training_args(args)

    model = get_peft_model(model, peft_config)
    
    print('trainer')
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        formatting_func=lambda example: formatting_prompts_func(example, tokenizer=tokenizer),
        max_seq_length=args.max_length,
    )
    print('training')
    trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    trainer.save_model(args.output_dir)
    
if __name__ == '__main__':
    main()
