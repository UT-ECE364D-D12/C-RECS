import numpy as np
import pandas as pd
import torch

from model.encoder import Encoder
from model.recommender import DeepFM
from utils.misc import cosine_distance


class Response:
    def __init__(self, encoder_path, recommender_path, device, dataset, offset):
        self.recommender = DeepFM()
        self.recommender = self.recommender.load_state_dict(
            torch.load(recommender_path, map_location=device)
        )
        self.data_embeddings = self.recommender["embedding.embedding.weight"][offset:]
        self.encoder = Encoder()
        self.encoder = self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=device)
        )
        self.dataset = dataset
        self.device = device

    def get_response(self, query, user_id, top_k=20):
        query_embedding = self.encoder(query)
        distances = cosine_distance(query_embedding, self.data_embeddings)
        _, top_k_indicies = torch.topk(distances, top_k, largest=False)

        # Get top k movie_id and movie_title pair from the dataset
        top_k_movies = self.dataset.iloc[top_k_indicies]
        top_k_movie_ids = top_k_movies["movie_id"].values

        # Make user_id and movie_id tensor pairs
        input_pairs = torch.tensor(
            [[user_id, movie_id] for movie_id in top_k_movie_ids]
        )
        input_pairs = input_pairs.to(self.device)

        # Get the predicted ratings for the top k movies
        ratings = self.recommender(input_pairs)
        ratings = ratings.squeeze(1)

        # Get the movie with the top rating
        top_movie_index = torch.argmax(ratings)
        top_movie_id = top_k_movie_ids[top_movie_index]
        top_movie_title = top_k_movies.iloc[top_movie_index]["movie_title"]

        # Craft the response
        response = self.__craft_response(top_movie_id, top_movie_title)

        return response

    def __craft_response(self, movie_id, movie_title):
        # Temprorary response, replace with LLM
        return f"The movie with the highest predicted rating is {movie_title} with movie_id {movie_id}."
