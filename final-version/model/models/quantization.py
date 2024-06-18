import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft-model-path', default="aerdna/testv0.1",help="Path to the fine-tuned model directory")
    parser.add_argument('--quantized-model-path', default="checkpoints/quantized_model",help="Path to save the quantized model")
    parser.add_argument('--quantization-bits', default=8, type=int, help="Number of bits for quantization (8, 4, or 2)")
    return parser.parse_args()


def quantize_model_(model, quantization_bits=8):
    if quantization_bits not in [8, 4]:
        raise ValueError("Quantization bits must be one of [8, 4]")

    # Apply quantization
    if quantization_bits == 8:
        model = bnb.nn.quantization.Quantizer.quantize_model(model, dtype=torch.float16)
    elif quantization_bits == 4:
        model = bnb.nn.quantization.Quantizer.quantize_model(model, dtype=torch.float16, blocksize=4)

    return model

def save_quantized_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized model saved to {output_dir}")

def main():
    args = parse_args()

    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_path)
    print(f"Model size: {model.get_memory_footprint():,} bytes")
    # Quantize the model
    from transformers import BitsAndBytesConfig

    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True
    )
    quantize_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, quantization_config=config)
    print(f"Model size: {quantize_model.get_memory_footprint():,} bytes")

    # Save the quantized model
    os.makedirs(args.quantized_model_path, exist_ok=True)
    quantize_model.save_pretrained(args.quantized_model_path)
    tokenizer.save_pretrained(args.quantized_model_path)
    print(f"Quantized model saved to {args.quantized_model_path}")
    quantize_model.push_to_hub("quantize_v0.2")
    print("done")
if __name__ == '__main__':
    main()
