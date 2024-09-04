import argparse
from typing import List, Callable
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizer, BitsAndBytesConfig, MistralForCausalLM
import logging

# parse arguments for path to dataset
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default="data/ml-100k/u.item", help='Path to the dataset')
parser.add_argument('--output_path', type=str, default="data/requests.h5", help='Path to save the requests')
args = parser.parse_args()
data_path = args.dataset_path
output_path = args.output_path

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
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    return model, tokenizer

def build_encoder():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.ERROR)
            
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").cuda()

    return model, tokenizer


prompt_generators = [
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to make a short request that will help the system to suggest the movie "{movie}" without mentioning its title, characters, or ANY plot elements. The response should instead use GENERAL characteristics like the genre, tone, and themes of the movie. Your request should be concise, sound conversational, and not be too enthusiastic. As an example, for the movie "Crazy Stupid Love" you should give a request like "I'm looking for a silly romantic comedy with a happy ending. Any suggestions?" Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to make a short request that will help the system to suggest the movie "{movie}" without mentioning its title, characters, or ANY plot elements. The response should instead use GENERAL characteristics like the genre, tone, and themes of the movie. Your request should be extremely short, sound conversational, not be too enthusiastic, and use informal wording. As an example, for the movie "La La Land" you should give a request like "I want to watch a rom com." Reply ONLY with the human-like request for a movie. DO NOT include any other text."""
]

class SimulatorDataset(Dataset):
    def __init__(self, movies: pd.DataFrame, tokenizer: AutoTokenizer, prompt_generators: List[Callable]) -> None:
        self.movies = movies
        self.tokenizer = tokenizer
        self.prompt_generators = prompt_generators

    def __len__(self):
        return len(self.movies) * len(self.prompt_generators)

    def __getitem__(self, idx: int):
        prompt_idx, movie_idx = divmod(idx, len(self.movies))
        movie_id, movie_title = self.movies.iloc[movie_idx][["movie_id", "movie_title"]]

        # Generate the prompt for the movie
        prompt = self.prompt_generators[prompt_idx](movie_title)

        # Form prompt
        chat = [{"role": "user", "content": prompt}]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        return movie_id, movie_title, prompt
    

SPLIT_STR = "[/INST] " # Minstral & Llama-2
def simulate(
    language_model: MistralForCausalLM,
    language_tokenizer: AutoTokenizer,
    encoder_model: BertModel,
    encoder_tokenizer: BertTokenizer,
    dataloader: DataLoader,
    max_length: int = 2048,
) -> pd.DataFrame:
    data = pd.DataFrame(columns=["movie_id", "movie_title", "request", "encoded_request"])

    with torch.no_grad():
        for movie_ids, movie_titles, prompts in tqdm(dataloader, desc="Simulating", unit="batch"):
            # Tokenize (llm)
            input_tokens = language_tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(language_model.device)

            # Generate request
            request_tokens = language_model.generate(**input_tokens, max_new_tokens=max_length, do_sample=True, pad_token_id=language_tokenizer.eos_token_id)

            # Decode request
            batch_requests = [language_tokenizer.decode(output, skip_special_tokens=True).split(SPLIT_STR)[-1] for output in request_tokens]

            # Tokenize (bert)
            encoder_tokens = encoder_tokenizer(batch_requests, padding=True, return_tensors="pt").to(language_model.device)

            # Encode request, grab the CLS token
            batch_encoded_requests = encoder_model(**encoder_tokens)

            encoded_requests = [hidden_state[0].cpu().numpy() for hidden_state in batch_encoded_requests.last_hidden_state]

            batch_requests = pd.DataFrame({
                "movie_id": movie_ids,
                "movie_title": movie_titles,
                "request": batch_requests,
                "encoded_request": encoded_requests
            })

            data = pd.concat([data, batch_requests], ignore_index=True)

    return data


# Load the model and tokenizer
language_model, language_tokenizer = build_language_model("mistralai/Mistral-7B-Instruct-v0.2")

encoder_model, encoder_tokenizer = build_encoder()

# ---------------------------Read in the dataset --------------------------------
columns = ["movie_id", "movie_title", "release_date", "url", "unknown",
           "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
           "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
           "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

with open(data_path, 'r', encoding='latin1') as file:
    first_line = file.readline().strip()
    split_line = first_line.split('|')

adjusted_columns = []
column_index = 0

for i, col in enumerate(split_line):
    if col == '':
        adjusted_columns.append(f'dummy_{i}')  # Insert a dummy column name
    else:
        adjusted_columns.append(columns[column_index])  # Use the provided column name
        column_index += 1

movies = pd.read_csv(data_path, sep='|', names=adjusted_columns, encoding='latin1', header=None)
movies = movies.drop(columns=[col for col in movies.columns if col.startswith('dummy_')])


# Create the dataset
dataset = SimulatorDataset(movies, language_tokenizer, prompt_generators)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# Simulate the responses
data = simulate(language_model, language_tokenizer, encoder_model, encoder_tokenizer, dataloader)

data.head()

# Save the new dataframe
data.to_hdf("data/requests.h5", key="df", mode="w", index=False)