# THIS FILE SHOULD BE RUN FROM THE MODEL DIRECTORY.

import json
from trl import DPOTrainer, DPOConfig
import transformers
from transformers import TrainingArguments
import argparse
import pandas as pd
from datetime import datetime
import torch
from peft import PeftModel, LoraConfig, TaskType
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
import wandb
from peft import get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/dpo/all_dpo.json')
    parser.add_argument('--ref-model-path', default=None)
    parser.add_argument('--beta', default=0.05, type=float)
    parser.add_argument('--n-epochs', default=3, type=int)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--eval-batch-size', default=2, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=4, type=int)
    parser.add_argument('--learning-rate', default=5e-4, type=float)
    parser.add_argument('--warmup-steps', default=100, type=int)
    parser.add_argument('--weight-decay', default=0.00, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=300, type=int)
    parser.add_argument('--logging-steps', default=10, type=int)
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--max-length', default=600, type=int)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--peft-config-r', default=8, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=16, type=int)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    parser.add_argument('--use-sft-model', default=False, type=bool)
    return parser.parse_args()

def map_data(example, bos_token, eos_token):
    question = example['prompt']
    if not question.startswith('Question: '):
        question = 'Question: ' + question

    return {
        'prompt': f"""{bos_token} [INST] ### Task: You will be asked a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. Your task is to answer the question to the best of your abilities. You must complete the task step by step and give your final answer by completing the following json: {'{'} 'Answer': ... {'}'}.\
                    You must explain your reasoning process.  Ensure that your answer is detailed, accurate, and logical, demonstrating a deep understanding of the topic. Remember, clarity, coherence, and accuracy are key components of a successful response.\
                        ### {question}\n### Answer: """ ,
        'chosen': example['chosen'] + f' {eos_token}',
        'rejected': example['rejected'] + f' {eos_token}'
    }


def count_trainable_parameters(model):
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"Trainable parameter: {name}, Shape: {param.shape}, Size: {param.numel()}")
    print(f"Total number of trainable parameters: {trainable_params}")
    return trainable_params


def main():
    args = parse_args()

    train_data = load_dataset('json', data_files=args.data_dir, split='train')
    test_data = load_dataset('json', data_files="datasets/dpo/epfl_test_data.json", split='train')
    print(f'Number of samples in the dataset: {len(train_data)}')
    print(f'Number of samples in the dataset: {len(test_data)}')
    ref_model_path = args.ref_model_path
    output_directory =f'{args.output_dir}/dpo_google-gemma-2b_{datetime.now()}'
    args.output_dir = output_directory.replace(' ', '_')

    # Load models and tokenizer
    if args.ref_model_path is not None:
        print("use sft model")
        ref_model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, device_map='auto',torch_dtype=torch.bfloat16)
        model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, is_trainable=True, device_map='auto',torch_dtype=torch.bfloat16)
        tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_path)
    else:
        model_name="google/gemma-2b"
        print(f"don't use sft model, using base: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',torch_dtype=torch.bfloat16)
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.peft_config_r,
                                 lora_alpha=args.peft_config_lora_alpha, lora_dropout=args.peft_config_lora_dropout)
        ref_model=None
        model = get_peft_model(model, peft_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token


    #model.enable_input_require_grads()
    train_data = train_data.map(lambda sample: map_data(sample, tokenizer.bos_token, tokenizer.eos_token))
    test_data = test_data.map(lambda sample: map_data(sample, tokenizer.bos_token, tokenizer.eos_token))
    print("Checking trainable parameters:")
    trainable_params = count_trainable_parameters(model)

    config = DPOConfig(
        # model_adapter_name='train_dpo',
        # ref_adapter_name='reference',
        output_dir=args.output_dir,
        max_steps=2000,
        overwrite_output_dir=False,
        num_train_epochs=args.n_epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = dict(use_reentrant=False),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        adam_epsilon=args.adam_epsilon,
        save_steps=args.save_steps,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        save_total_limit=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        evaluation_strategy='steps',
        eval_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        bf16=True
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        beta=args.beta,
        args=config,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        max_length=1500,
        is_encoder_decoder=False,

    )


    # Start training
    dpo_trainer.train()

    print("SAVING MODEL at ", args.output_dir)

    with open(args.output_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    dpo_trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
