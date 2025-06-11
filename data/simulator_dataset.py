from typing import Callable, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PromptGenerator = Callable[[str], str]


class SimulatorDataset(Dataset):
    """
    Dataset for the user simulator.

    Args:
        items (pd.DataFrame): The items dataframe.
        tokenizer (AutoTokenizer): The tokenizer.
        prompt_generators (List[Callable]): The prompt generators which generate prompts given an item title.
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
            idx (int): The index of the item.

        Returns:
            item_id (int): The item ID.
            item_title (str): The item title.
            prompt (str): The prompt.
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
        model (AutoModelForCausalLM): Language model.
        tokenizer (AutoTokenizer): Tokenizer.
        dataloader (DataLoader): The data to generate responses for.
        max_length (int, optional): The maximum length of the generated response.

    Returns:
        data (pd.DataFrame): The generated responses.
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
