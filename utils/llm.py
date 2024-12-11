import logging

logging.getLogger("bitsandbytes").setLevel(logging.CRITICAL)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_language_model(model_name: str = "google/gemma-7b-it") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True, 
        quantization_config=config,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        
    return model, tokenizer