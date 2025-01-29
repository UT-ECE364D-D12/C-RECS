import pandas as pd
from imdb import Cinemagoer
from tqdm import tqdm

links = pd.read_csv("data/ml-20m/links.csv", header=0, names=["item_id", "imdb_id", "tmdb_id"])

imdb = Cinemagoer()


def get_description(imdb_id: str):
    try:
        movie = imdb.get_movie(imdb_id)

        if "plot outline" in movie.keys():
            return movie["plot outline"]
        elif "plot" in movie.keys():
            return movie["plot"]

        return None
    except:
        return None


tqdm.pandas(desc="Fetching Descriptions", unit="movie")

links["description"] = links["imdb_id"].progress_apply(get_description)

descriptions = links[["item_id", "description"]]

movies = pd.read_csv("data/ml-20m/movies.csv", header=0, names=["item_id", "item_title", "genres"])

movies = movies[["item_id", "item_title"]]

descriptions = descriptions.merge(movies, on="item_id")[["item_id", "item_title", "description"]]

# If a movie has no description, we will use the title as the description
descriptions["description"] = descriptions["description"].fillna(descriptions["item_title"])

descriptions.to_csv("data/ml-20m/descriptions.csv", index=False)
