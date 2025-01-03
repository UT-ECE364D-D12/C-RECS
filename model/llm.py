from utils.misc import suppress_warnings

suppress_warnings()

from importlib.util import find_spec

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_language_model(model_name: str = "google/gemma-7b-it", quantize: str = "8bit") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # If bitsandbytes is not installed, quantization is not possible.
    if find_spec("bitsandbytes") is None:
        quantize = None

    if quantize is None:
        quantization_config = None
    elif quantize == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantize == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError("Invalid quantization type. Choose between 'None', '4bit', and '8bit'.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # Set the padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        
    return model, tokenizer