from importlib.util import find_spec
from typing import Callable, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PromptGenerator = Callable[[str], str]


class SimulatorDataset(Dataset):
    """
    Dataset for the user simulator.

    Args:
        items: The items dataframe.
        tokenizer: The tokenizer.
        prompt_generators: The prompt generators which generate prompts given an item title.
    """

    def __init__(self, items: pd.DataFrame, tokenizer: AutoTokenizer, prompt_generators: List[PromptGenerator]) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.prompt_generators = prompt_generators

    def __len__(self) -> int:
        return len(self.items) * len(self.prompt_generators)

    def __getitem__(self, idx: int) -> Tuple[int, str, str]:
        """
        Retrieves an item and associated prompt.

        Args:
            idx: The index of the item.

        Returns:
            item_id: The item ID.
            item_title: The item title.
            prompt: The prompt.
        """

        prompt_idx, item_idx = divmod(idx, len(self.items))
        item_id, item_title = self.items.iloc[item_idx][["item_id", "item_title"]]

        # Generate the prompt for the movie
        prompt = self.prompt_generators[prompt_idx](item_title)

        # Form prompt
        chat = [{"role": "user", "content": prompt}]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        return item_id, item_title, prompt


def simulate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    max_length: int = 64,
) -> pd.DataFrame:
    """
    Generate responses for a given prompt using a model.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        dataloader: The data to generate responses for.
        max_length: The maximum length of the generated response.

    Returns:
        data: The generated responses.
    """
    data = pd.DataFrame(columns=["item_id", "item_title", "text"])

    with torch.no_grad():
        for item_ids, item_titles, prompts in tqdm(dataloader, desc="Simulating", unit="batch", dynamic_ncols=True):
            # Tokenize input
            batch_input_tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

            # Generate response
            batch_output_tokens = model.generate(**batch_input_tokens, max_new_tokens=max_length, do_sample=True)

            # Decode response
            responses = [
                tokenizer.decode(output_tokens[len(input_tokens) :], skip_special_tokens=True).strip('"')
                for input_tokens, output_tokens in zip(batch_input_tokens["input_ids"], batch_output_tokens)
            ]

            batch_output = pd.DataFrame(
                {
                    "item_id": item_ids,
                    "item_title": item_titles,
                    "text": responses,
                }
            )

            data = pd.concat([data, batch_output], ignore_index=True)

    return data


def build_language_model(model_name: str = "google/gemma-7b-it", quantize: str = "8bit") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and potentially quantize a language model.

    Args:
        model_name: Pre-trained model name
        quantize: Quantization type

    Returns:
        model: Language model
        tokenizer: Tokenizer
    """

    # If bitsandbytes is not installed, quantization is not possible.
    quantize = None if find_spec("bitsandbytes") is None else quantize

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

    # Use Flash Attention if it is installed
    attn_implementation = None if find_spec("flash_attn") is None else "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
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
