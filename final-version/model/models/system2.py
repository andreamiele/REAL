import pandas as pd
import numpy as np
import os
from collections import namedtuple
import torch
from tqdm import tqdm
import warnings
import json
import time
import transformers
import sys
import pathlib
from transformers import AutoModelForCausalLM
import argparse
from peft import AutoPeftModelForCausalLM, PeftModel
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import re
from jsonformer import Jsonformer
import joblib

sys.path.append('src/')
warnings.filterwarnings("ignore")
#'max_new_tokens':10,

def print_memory_usage(device):
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    print(f"Memory allocated: {allocated:.2f} GB")
    print(f"Memory reserved: {reserved:.2f} GB")


def extract_generated_answer(y):
    """
    Extract the first capital letter in the next 20 characters after splitting the text on "The correct".
    If no capital letter is found, return an empty string.
    """
    parts = y.split("The correct Answer")
    #print(f"Split parts: {parts}")  # Debug print to check the parts
    for i, part in enumerate(parts):
        if i!=0:
        # Look for the first capital letter in the next 20 characters
            match = re.search(r'[A-Z]', part[:20])
            if match:
                return match.group(0)

    return ""

def compute_and_save_embeddings(docs, encoder, tokenizer, filename='embeddings.joblib'):
    embeddings = []
    for example in tqdm(docs):
        text = example["text"]
        with torch.no_grad():  # No need to track gradients for embeddings
            embedding = encoder(**tokenizer(text, return_tensors="pt", max_length=512, truncation=True)).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    joblib.dump(embeddings, filename)
    print(f"Embeddings saved to {filename}")
def batch_generate(prompts, model, tokenizer, **generate_kwargs):
    tokenized_prompts = tokenizer(prompts, return_tensors='pt', max_length=1024, truncation=True).to(model.device)

    with torch.no_grad():
        output = model.generate(**tokenized_prompts,
                                **generate_kwargs,
                                pad_token_id=tokenizer.eos_token_id)
    output_decoded = tokenizer.batch_decode(output, skip_special_tokens=False)
    return output_decoded

def generate(prompt: str, model, tokenizer, **generate_kwargs):
    return batch_generate(prompt, model, tokenizer, **generate_kwargs)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path')

    parser.add_argument('--print-first-n', type=int, default=100,
                        help='Print and exit after processing the first N samples')
    parser.add_argument('--mode', default="pt",
                        help='Print and exit after processing the first N samples')
    parser.add_argument('--use-rag', default=False,
                        help='Print and exit after processing the first N samples')
    parser.add_argument('--use-quantize', default=False,
                        help='Print and exit after processing the first N samples')
    parser.add_argument('--use-epfl', default=False,
                        help='Print and exit after processing the first N samples')
    parser.add_argument('--merge', default=False,
                        help='Print and exit after processing the first N samples')
    parser.add_argument('--baseline', default=False,
                        help='Print and exit after processing the first N samples')
    parser.add_argument('--model-name', default="",
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

    use_rag = args.use_rag

    if use_rag:
        torch.set_grad_enabled(False)
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        dpr_ctx_encoder = 'facebook/dpr-ctx_encoder-single-nq-base'
        dpr_question_encoder = "facebook/dpr-question_encoder-single-nq-base"
        ctx_encoder = DPRContextEncoder.from_pretrained(dpr_ctx_encoder)
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_ctx_encoder)

        docs = load_dataset('json', data_files='documents/retrieved_documents_with_mapping.json', split='train')
        #docs = docs.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example["text"], return_tensors="pt", max_length=512))[0][0].numpy()})
        #print(len(docs))
        docs.add_faiss_index(column='embeddings')

        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
        question_encoder = DPRQuestionEncoder.from_pretrained(dpr_question_encoder)
        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_question_encoder)

    use_quantize = args.use_quantize

    if use_quantize:
        model_name = "microsoft/phi-2"
        if args.model_name != "":
            model_name = args.model_name
        config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=config,cache_dir='cache')
    else:
        if args.merge:
            model_name = "microsoft/phi-2"
            model_name = "google/gemma-2b"
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', cache_dir='cache')
            model = PeftModel.from_pretrained(model, "aerdna/gemma-dpo-stem-real", device_map='auto', cache_dir='cache')
            model = model.merge_and_unload()
            print("merge and unload")
            model.to(device)
        else:
            if args.baseline:
                #model_name = "microsoft/phi-2"
                model_name="google/gemma-2b"
                if args.model_name!="":
                    model_name = args.model_name
                model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
                model.to(device)
            else:
                model_name = 'checkpoints/changed_gemma-dpo-stem-real'
                #model_name= "checkpoints/20k_gemma-dpo-stem-real"
                model_name ="checkpoints/spe_phi-2"
                model_name = "checkpoints/dpo_phi-2_2024-06-12_15:22:11.187752" # DPO PHI 10K MIXED DATA
                model_name = "checkpoints/dpo_phi-2_full_epfl_2024-06-12_22:35:16.778027" # DPO PHI FULL EPFL
                #model_name = "aerdna/gemma-dpo-stem-real"
                #model_name = "checkpoints/20k_gemma-dpo-stem-real"
                if args.model_name!="":
                    model_name = args.model_name
                model = AutoPeftModelForCausalLM.from_pretrained(model_name, cache_dir='cache')
                model.to(device)

    print(f"Using model: {model_name}")
    print(args)
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,max_length=3000)
    tokenizer.pad_token = tokenizer.eos_token
    GENERATION_KWARGS = {
        "max_new_tokens": 500,
        "num_return_sequences": 1,
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 0.9,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 2
    }

    epfl_data= args.use_epfl
    if epfl_data:
        test_set = pd.read_json('datasets/mcq/mcq_epfl_expl_letter_final.jsonl', lines=True)[550:]
        test_set = Dataset.from_pandas(test_set)
    else:
        #dataset = pd.read_parquet('datasets/mcq/train-00000-of-00001.parquet')
        #dataset = load_dataset('thewordsmiths/stem_mcqa')
        # Extract the train dataset
        #df = pd.DataFrame(dataset['train'])[:2000]
        # Prepare the dataset for training
        # test_set = df[['question', 'options', 'solution', 'answer']]
        #test_set = Dataset.from_pandas(dataset)
        dataset = load_dataset('mvujas/stem_mcqa_questions', split='train[:2000]')
        # Extract the train dataset
        test_set = dataset

    answers = []
    correct_answers = 0
    total_options = 0
    memory=0
    timer=0
    nb=50
    for i, entry in tqdm(enumerate(test_set), total=nb):
        if i >= nb:
            break
        question = entry['question'].split("Question: ")[-1]
        ground_truth = entry['answer'].split("###The correct answer is letter: ")[-1].strip()[0]
        options_start = question.split("\n\nOptions:\n")[-1]
        only_question = question.split("\n\nOptions:\n")[0]
        options_str = question.strip()

        if epfl_data:
            option_lines = re.findall(r'^[A-Z]\..*$', options_str, re.MULTILINE)
            num_options = len(option_lines)
            total_options += num_options


        else:
            option_lines = question.split("Options:")[-1]
            only_question = question.split("Options:")[0]
            #option_lines = "\n".join([f"{chr(65 + idx)}. {opt.strip()}" for idx, opt in enumerate(entry['options'])])

        if use_rag:
            question_embedding = question_encoder(**question_tokenizer(question, return_tensors="pt", max_length=512, truncation=True))[0][0].numpy()
            _, retrieved_examples = docs.get_nearest_examples('embeddings', question_embedding, k=1)

            retrieved = list(set(retrieved_examples['text']))[:1]#to add somewhere in the prompt
            retrieved_context = " ".join(retrieved)
            prompt = f"""Context: {retrieved} {tokenizer.bos_token} [INST] ### Task: You will be asked a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. Your task is to answer the question to the best of your abilities. You must complete the task step by step and precise what is your final answer.
                Remember, clarity, coherence, and accuracy are key components of a successful response.
                ### {only_question}.\n ### Options: {option_lines}. Use a capital letter.[/INST] \n### Answer: """
        else:
            prompt = f"""{tokenizer.bos_token} [INST] ### Task: You will be asked a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. Your task is to answer the question to the best of your abilities. You must complete the task step by step and precise what is your final answer.
                    Remember, clarity, coherence, and accuracy are key components of a successful response.\
                    ### {only_question}.\n ### Options: {option_lines}. Use a capital letter.[/INST] \n ### Answer: """
        #print(prompt)
        start_time = time.time()
        ys = generate(prompt, model, tokenizer, **GENERATION_KWARGS)

        split=ys[0][len(prompt):len(prompt)+100].split("The correct option is ")[-1]

        split_phi3=ys[0][len(prompt):len(prompt)+40].split("The correct option is ")[-1][4:]
        print(ys[0][len(prompt):])
        prompt2 = f"""The correct letter is : {split}.
                \n Generate the correct letter based on the following schema:"""
        #print("---")
        #print(prompt2)
        json_schema = {
            "type": "object",
            "properties": {
                "correct_letter": {"type": "string"}
            },
            "required": ["correct_option"]
        }

        jsonformer = Jsonformer(model, tokenizer, json_schema, prompt2)
        response = jsonformer()
        #print(response)

        elapsed_time = time.time() - start_time
        timer+=elapsed_time
        memory+=torch.cuda.memory_allocated(device) / 1024 ** 3

        #print(f"Time taken for generation: {elapsed_time:.2f} seconds")
        #print_memory_usage(device)
        generated_answer = response.get('correct_letter', '')
        match = re.search(r'[A-Za-z]', generated_answer)
        if match:
            generated_answer = match.group(0).upper()
        else:
            generated_answer = ''
        output = f"==> Ground Truth {i + 1}: {ground_truth} and generated answer: {generated_answer}"
        if generated_answer == ground_truth:
            correct_answers += 1
            output=" ✅ "+output
        else:
            output =" ❌ "+output


        print(output)
        answers.append({'question': prompt, 'policy': generated_answer})
    mean_options = total_options / nb
    print(f"Score: {correct_answers/nb}")
    #print(f"Mean number of options: {mean_options}")
    #print(f"If random: {1/mean_options}")
    print(f"Average memory used: {memory/nb:.2f} GB")
    print(f"Average time taken: {timer/nb:.2f} seconds")
    output_dir = f'datasets/inference/'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_dir + f'generations_mcq_{args.mode}.json', 'w') as f:
        for item in answers:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
