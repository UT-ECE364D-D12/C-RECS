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

ratings.to_csv("data/ml-20m/ratings.csv", index=False)

movies.to_csv("data/ml-20m/movies.csv", index=False)