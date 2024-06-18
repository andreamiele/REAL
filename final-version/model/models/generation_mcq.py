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
import argparse
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
import pandas as pd
import json
import sys
import os
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
from transformers import BitsAndBytesConfig
from tqdm import tqdm
sys.path.append('src/')
warnings.filterwarnings("ignore")

GENERATION_KWARGS = {'max_length': 800, 'no_repeat_ngram_size': 2, 'do_sample': True, 'top_p': 0.75}

def print_memory_usage(device):
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"Memory allocated: {allocated:.2f} GB")
    print(f"Memory reserved: {reserved:.2f} GB")

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

    parser.add_argument('--model-path')

    parser.add_argument('--print-first-n', type=int, default=100,
                        help='Print and exit after processing the first N samples')
    parser.add_argument('--mode', default="base",
                        help='Print and exit after processing the first N samples')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    use_quantize = False 

    if use_quantize:
        model_name = "aerdna/quantize_v0.2"
    else: 
        model_name = 'aerdna/testv0.1'

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cachewfwfe')
    #model.to(device)
    model.eval()

    ### SFT and DPO have the same tokenizer -- but not sure for the other two
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


    test_set = pd.read_json('datasets/mcq/mcq_epfl_expl_letter.jsonl', lines=True)[:650]
    test_set = Dataset.from_pandas(test_set)
    answers = []

    for i, entry in tqdm(enumerate(test_set), total=10):
        if i >= 10:
            break


        question = entry['question']
        
        prompt = f"""{tokenizer.bos_token} [INST] ### Task: You will be asked a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. Your task is to answer the question to the best of your abilities. You must complete the task step by step and give your final answer by completing the following json: {'{'} 'Answer': ... {'}'}.\
                    You must explain your reasoning process.  Ensure that your answer is detailed, accurate, and logical, demonstrating a deep understanding of the topic. Remember, clarity, coherence, and accuracy are key components of a successful response. You must give explanations starting with "Explanations:" and then give the answer with the LETTER with "///Answer:" For instance, if the answer is A-False, you will put ///Answer: A \
                        ### {question}\n### Answer:"""

        start_time = time.time()
        ys = generate(prompt, model, tokenizer, **GENERATION_KWARGS)
        elapsed_time = time.time() - start_time
        print(f"Time taken for generation: {elapsed_time:.2f} seconds")
        print_memory_usage(device)
        ys = map(lambda y: y.split('### Answer: ')[-1].strip(), ys)
        for y in zip(ys):
            #print(f"Generated Answer {i + 1}: {y}")
            answers.append({'question': prompt, 'policy': y})

    output_dir = f'datasets/inference/'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_dir + f'generations_mcq_{args.mode}.json', 'w') as f:
        for item in answers:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
