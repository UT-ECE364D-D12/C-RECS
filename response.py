import numpy as np
import pandas as pd
import torch

from model.encoder import Encoder
from model.recommender import DeepFM
from utils.misc import cosine_distance

import json
from llamaapi import LlamaAPI


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
        llama = LlamaAPI("api_token")

        api_request_json = {
            "messages": [
                {"role": "user", "content": "Can you recommend me a movie?"}
            ],
            "functions": [
                {
                    "name": "get_movie_recommendation",
                    "description": "Recommend a movie based on the user's preferences",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "genre": {
                                "type": "string",
                                "description": "The genre of the movie, e.g. Action, Romance, Comedy, Sci-Fi",
                            },
                            "year": {
                                "type": "number",
                                "description": "The year of the movie's release, e.g. 1994, 2020",
                            },
                            "title": {
                                "type": "string",
                                "description": "The title of the movie to recommend",
                            },
                            "description": {
                                "type": "string",
                                "description": "A brief description of the movie",
                            }
                        },
                    },
                    "required": ["title", "description"]
                }
            ],
            "stream": False,
            "function_call": "get_movie_recommendation",
        }

        response = llama.run(api_request_json)

        response_json = response.json()

        movie_title = response_json['movie']['title']
        movie_description = response_json['movie']['description']

        recommendation = f"If you're in the mood for a great film, {movie_title} is a must-watch. {movie_description} Ready to dive into a fantastic movie experience?"

        print(recommendation)
        return f"The movie with the highest predicted rating is {movie_title} with movie_id {movie_id}."
