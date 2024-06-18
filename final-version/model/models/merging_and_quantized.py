import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import bitsandbytes as bnb
import os
import argparse
from peft import AutoPeftModelForCausalLM, PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default="microsoft/phi-2",help="Path to the fine-tuned model directory")
    parser.add_argument('--peft-model', default="checkpoints/dpo_phi-2_full_epfl_2024-06-12_22:35:16.778027",help="Path to save the quantized model")
    parser.add_argument('--full-model-path', default="checkpoints/",help="Path to the fine-tuned model directory")
    parser.add_argument('--quantized-model-path', default="checkpoints/phi2-q",help="Path to save the quantized model")
    parser.add_argument('--quantized-name', default="phi2-q",help="Path to the fine-tuned model directory")
    return parser.parse_args()

def main():
    args = parse_args()

    q = False
    if args.peft_model!="":
        print("Merging a peft model")
        print(args.base_model)
        print(args.peft_model)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', cache_dir='cache')
        tokenizer=AutoTokenizer.from_pretrained(args.base_model, cache_dir='cache')
        model = PeftModel.from_pretrained(model, args.peft_model, device_map='auto', cache_dir='cache')
        model = model.merge_and_unload()
        model.save_pretrained(args.full_model_path)
        tokenizer.save_pretrained(args.full_model_path)
        print("Merged saved locally")
        if q:
            config = BitsAndBytesConfig(load_in_8bit=True)
            q_model = AutoModelForCausalLM.from_pretrained(args.full_model_path, quantization_config=config, cache_dir='cache')
            q_model.save_pretrained(args.quantized_model_path)
            tokenizer.save_pretrained(args.quantized_model_path)
            print("Saved locally")
            q_model.push_to_hub(args.quantized_name)
            tokenizer.push_to_hub(args.quantized_name)
            print("Quantized model pushed to hub")
        else:
            model.push_to_hub("phi2-dpo")
            tokenizer.push_to_hub("phi2-dpo")
            print("Not quantized model pushed to hub")
    else:
        print("Not a PEFT model")
        config = BitsAndBytesConfig(load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir='cache')
        q_model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=config,
                                                       cache_dir='cache')
        q_model.save_pretrained(args.quantized_model_path)
        tokenizer.save_pretrained(args.quantized_model_path)
        print("Saved locally")
        q_model.push_to_hub(args.quantized_name)
        tokenizer.push_to_hub(args.quantized_name)
        print("Pushed to hub")

if __name__ == '__main__':
    main()
