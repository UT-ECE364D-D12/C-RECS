import numpy as np
import pandas as pd
import torch

from model.encoder import Encoder
from model.recommender import DeepFM
from utils.data import get_feature_sizes
from utils.misc import build_language_model, cosine_distance


class Response:
    def __init__(
        self,
        encoder_path: str,
        recommender_path: str,
        device: str,
        item_data: pd.DataFrame,
        rating_data: pd.DataFrame,
        offset: int,
    ):
        self.recommender = (
            DeepFM(
                feature_dims=get_feature_sizes(rating_data),
                embed_dim=768,
                mlp_dims=(16, 16),
                dropout=0.8,
            )
            .to(device)
            .eval()
        )
        self.recommender.load_state_dict(
            torch.load(recommender_path, map_location=device)
        )
        self.data_embeddings = torch.load(recommender_path, map_location=device)[
            "embedding.embedding.weight"
        ][offset:]
        self.encoder = Encoder().to(device).eval()
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.item_data = item_data
        self.device = device

    def get_response(self, query: str, user_id: int, top_k: int = 20, **kwargs):
        # Get the top k items
        query_embedding = self.encoder(query)
        distances = cosine_distance(query_embedding, self.data_embeddings)
        _, top_k_indicies = torch.topk(distances, top_k, largest=False)
        top_k_indicies = top_k_indicies.cpu().numpy()

        # Get top k items and id pair from the dataset
        top_k_items = self.item_data.iloc[top_k_indicies]
        top_k_ids = top_k_items["movie_id"].values

        # Make user_id and item_id tensor pairs
        input_pairs = torch.tensor([[user_id, item_id] for item_id in top_k_ids])
        input_pairs = input_pairs.to(self.device)

        # Get the predicted ratings for the top k items
        ratings = self.recommender(input_pairs)

        # Get the item with the top rating
        top_item_index = torch.argmax(ratings).item()
        top_item_name = top_k_items.iloc[top_item_index]["movie_title"]

        # Craft the response
        response = self.__craft_response(query, top_item_name, **kwargs)

        return response

    def __craft_response(
        self, query, item_name: str, model_name: str, split_string: str
    ) -> str:

        # Build the language model
        model_name = model_name
        language_model, language_tokenizer = build_language_model(model_name=model_name)

        # Create a prompt for the item
        chat = [
            {
                "role": "user",
                "content": f"You are a response generator that generates responces to queries for recommendations. I will give you a query and a recommendation, and you will generate a response based on the query. Please reply with only the response, no preamble, and no quotations on the response. The user query was: '{query}'. Create a response with the item '{item_name}'. Make the response friendly and don't include any spoilers.",
            },
        ]
        prompt = language_tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Tokenize the prompt
        input_tokens = language_tokenizer(
            prompt, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(language_model.device)

        # Generate the request
        max_length = 2048
        request_tokens = language_model.generate(
            **input_tokens,
            max_new_tokens=max_length,
            do_sample=True,
            pad_token_id=language_tokenizer.eos_token_id,
        )

        # Decode the request
        request = language_tokenizer.decode(
            request_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).split(split_string)[-1]

        return request


if __name__ == "__main__":
    # Load the dataset
    item_data = pd.read_csv(
        "data/ml-100k/u.item",
        sep="|",
        names=[
            "movie_id",
            "movie_title",
            "release_date",
            "dummy",
            "url",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ],
        encoding="latin1",
        header=None,
    )

    # Load the rating data
    rating_data = pd.read_csv(
        "data/ml-100k/u.data",
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    # Load the response model
    response = Response(
        encoder_path="weights/encoder/encoder.pt",
        recommender_path="weights/recommender/deepfm.pt",
        device="cuda",
        item_data=item_data,
        rating_data=rating_data,
        offset=943,
    )

    # Get the response
    query = "I want to watch a romantic comedy."
    user_id = 1
    model_name = "google/gemma-7b-it"
    split_string = "\nmodel\n"
    response = response.get_response(
        query,
        user_id,
        top_k=20,
        model_name=model_name,
        split_string=split_string,
    )
    print(response)
