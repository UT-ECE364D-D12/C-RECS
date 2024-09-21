
import pandas as pd

movies = pd.read_csv("data/ml-20m/movies.csv", header=0, names=["movie_id", "movie_title", "genres"])

ratings = pd.read_csv("data/ml-20m/ratings.csv", header=0, names=["user_id", "movie_id", "rating", "timestamp"]).astype(int)

movies = movies[movies["movie_id"].isin(ratings["movie_id"])]

num_movie_ratings = ratings.groupby("movie_id").size()

movies = movies[movies["movie_id"].isin(num_movie_ratings[num_movie_ratings >= 5].index)]

ratings = ratings[ratings["movie_id"].isin(movies["movie_id"])]

movies.to_csv("data/ml-20m/movies.csv", index=False)

ratings.to_csv("data/ml-20m/ratings.csv", index=False)
