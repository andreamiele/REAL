import pandas as pd
import numpy as np
import os
from collections import namedtuple
import torch
from tqdm import tqdm
import warnings
import json
import time
import os
import transformers
import sys
import pathlib
from transformers import AutoModelForCausalLM
from model_dpo import AutoDPOModelForCausalLM
import argparse
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset

sys.path.append('src/')
warnings.filterwarnings("ignore")

GENERATION_KWARGS = {'max_length': 1500, 'no_repeat_ngram_size': 2, 'do_sample': True, 'top_p': 0.75}


def batch_generate(prompts, model, tokenizer, num_return_sequences, **generate_kwargs):

    tokenized_prompts = tokenizer(prompts, return_tensors='pt', max_length=1024, truncation=True).to(model.device)

    with torch.no_grad():
        output = model.generate(**tokenized_prompts,
                                **generate_kwargs,
                                num_return_sequences=num_return_sequences,
                                pad_token_id=tokenizer.eos_token_id)
    output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    return output_decoded


def generate(prompt: str, model, tokenizer, **generate_kwargs):
    return batch_generate(prompt, model, tokenizer, num_return_sequences=1, **generate_kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', required=True)

    parser.add_argument('--ref-model-path', required=True)
    parser.add_argument('--print-first-n', type=int, default=100,
                        help='Print and exit after processing the first N samples')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir='cachewfwfe')
    model.to(device)
    model.eval()

    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_path, cache_dir='cachewfwfe')
    ref_model.to(device)
    ref_model.eval()

    ### SFT and DPO have the same tokenizer -- but not sure for the other two
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)

    with open('data/dpo_prepared/dpo_test.json', 'r') as f:
        test_set = json.load(f)

    test_set = test_set[:1500:15]
    answers = []

    for i, entry in enumerate(test_set):
        if i >= args.print_first_n:
            break


        question = entry['prompt']
        
        prompt = f"""{tokenizer.bos_token} [INST] ### Task: You will be asked a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. Your task is to answer the question to the best of your abilities. You must complete the task step by step and give your final answer by completing the following json: {'{'} 'Answer': ... {'}'}.\
                        You must explain your reasoning process.  Ensure that your answer is detailed, accurate, and logical, demonstrating a deep understanding of the topic. Remember, clarity, coherence, and accuracy are key components of a successful response.\
                            ### {question}\n### Answer: """

        start_time = time.time()
        ys = generate(prompt, model, tokenizer, **GENERATION_KWARGS)

        ys_ref = generate(prompt, ref_model, tokenizer, **GENERATION_KWARGS)
        elapsed_time = time.time() - start_time
        print(f"Time taken for generation: {elapsed_time:.2f} seconds")

        ys = map(lambda y: y.split('### Answer: ')[-1].strip(), ys)
        ys_ref = map(lambda y: y.split('### Answer: ')[-1].strip(), ys_ref)
        for y,y_ref in zip(ys, ys_ref):
            #print(f"Generated Answer {i + 1}: {y}")
            answers.append({'question': prompt, 'policy': y, 'reference': y_ref})

    output_dir = f'data/inference/'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_dir + f'generations.json', 'w') as f:
        for item in answers:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
