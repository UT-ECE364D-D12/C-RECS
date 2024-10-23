# Preprocess the MovieLens 20M dataset

import os

import numpy as np
import pandas as pd

from utils.misc import set_random_seed

set_random_seed(42)

movies = pd.read_csv("data/ml-20m/movies.csv", header=0, names=["movie_id", "movie_title", "genres"])

ratings = pd.read_csv("data/ml-20m/ratings.csv", header=0, names=["user_id", "movie_id", "rating", "timestamp"]).astype({"user_id": int, "movie_id": int, "rating": float, "timestamp": int})

# Remove movies that have no ratings
movies = movies[movies["movie_id"].isin(ratings["movie_id"])]

# Remove duplicate movies
ratings = pd.merge(ratings, movies[["movie_id", "movie_title"]], on="movie_id")

ratings = ratings[["user_id", "movie_title", "rating", "timestamp"]]

movies = movies.drop_duplicates("movie_title")

ratings = pd.merge(ratings, movies, on="movie_title")

ratings = ratings[["user_id", "movie_id", "rating", "timestamp"]]

# Remove users that have rated less than 20 movies
num_user_ratings = ratings.groupby("user_id").size()

ratings = ratings[ratings["user_id"].isin(num_user_ratings[num_user_ratings >= 20].index)]

# Sample 2000 users
users = np.random.choice(ratings["user_id"].unique(), 2000, replace=False)

ratings = ratings[ratings["user_id"].isin(users)]

movies = movies[movies["movie_id"].isin(ratings["movie_id"])]

# Convert user_id and movie_id to unique_id
user_id_to_unique_id = {user_id: i for i, user_id in enumerate(ratings["user_id"].unique())}
item_id_to_unique_id = {movie_id: i for i, movie_id in enumerate(movies["movie_id"].unique())}

if os.path.exists("data/ml-20m/requests.csv"):
    requests = pd.read_csv("data/ml-20m/requests.csv")
    
    requests = requests[requests["movie_id"].isin(movies["movie_id"])]

    requests["movie_id"] = requests["movie_id"].map(item_id_to_unique_id)

    requests.to_csv("data/ml-20m/requests.csv", index=False)

if os.path.exists("data/ml-20m/descriptions.csv"):
    descriptions = pd.read_csv("data/ml-20m/descriptions.csv")

    descriptions = descriptions[descriptions["movie_id"].isin(movies["movie_id"])]

    descriptions["movie_id"] = descriptions["movie_id"].map(item_id_to_unique_id)

    descriptions.to_csv("data/ml-20m/descriptions.csv", index=False)

movies["movie_id"] = movies["movie_id"].map(item_id_to_unique_id)

movies.to_csv("data/ml-20m/movies.csv", index=False)

ratings["user_id"] = ratings["user_id"].map(user_id_to_unique_id)
ratings["movie_id"] = ratings["movie_id"].map(item_id_to_unique_id)

ratings.to_csv("data/ml-20m/ratings.csv", index=False)


# Process ratings
grouped_ratings = ratings.sort_values(by=["user_id", 'timestamp']).groupby("user_id").agg({
    "movie_id": list,
    "rating": list,
    "timestamp": list,
}).reset_index()

no_movie_id = ratings["movie_id"].nunique()

processed_ratings = []

for user_id, item_ids, item_ratings, timestamps in grouped_ratings.values:
    feature_ids = [no_movie_id]
    feature_ratings = [5.0]

    for item_id, rating, timestamp in zip(item_ids, item_ratings, timestamps):
        processed_ratings.append({
            "user_id": user_id,
            "feature_ids": feature_ids.copy(),
            "feature_ratings": feature_ratings.copy(),
            "item_id": item_id,
            "rating": rating,
            "timestamp": timestamp,
        })
        
        feature_ids.append(item_id)
        feature_ratings.append(rating)

processed_ratings = pd.DataFrame(processed_ratings)

processed_ratings.to_hdf("data/ml-20m/processed_ratings.hdf", key="data", mode="w", index=False)