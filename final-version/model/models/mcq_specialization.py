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
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import wandb
import os

def formatting_prompts_func(example, tokenizer):
    data = []
    for i in range(len(example['question'])):
        instruction = example['question'][i]
        completion = example['answer'][i]
        if not instruction.startswith('Question: '):
            instruction = 'Question: ' + instruction
        prompt = f"""{tokenizer.bos_token} [INST] ### Task: You will be asked a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. Your task is to answer the question to the best of your abilities. You must complete the task step by step and give your final answer by completing the following json: {'{'} 'Answer': ... {'}'}.\
                    You must explain your reasoning process.  Ensure that your answer is detailed, accurate, and logical, demonstrating a deep understanding of the topic. Remember, clarity, coherence, and accuracy are key components of a successful response. You must give explanations starting with "Explanations:" and then give the answer with the LETTER with "///Answer:" For instance, if the answer is A-False, you will put ///Answer: A \
                        ### {instruction}\n### Answer: {completion} {tokenizer.eos_token}"""
        data.append(prompt)
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo-model-path', default='aerdna/gemma-dpo-stem-real')
    parser.add_argument('--n-epochs', default=4, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--eval-steps', default=80, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=2e-4, type=float)
    parser.add_argument('--warmup-steps', default=0, type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=80, type=int)
    parser.add_argument('--logging-steps', default=20, type=int)
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--max-length', default=1500, type=int)
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
    model_path = args.dpo_model_path

    file_path = 'datasets/mcq/mcq_epfl_expl_letter.jsonl'
    train_data = pd.read_json(file_path, lines=True)[:650]
    train_data = Dataset.from_pandas(train_data)

    output_directory =f'{args.output_dir}/sft_{model_path.split("/")[-1]}'
    args.output_dir = output_directory.replace(' ', '_')

    model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map='auto',is_trainable=True, cache_dir='cache')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('model loaded')
    
    training_args = get_training_args(args)
    
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
    trainer.merge_and_unload()
    trainer.save_model(args.output_dir)
    
if __name__ == '__main__':
    main()
