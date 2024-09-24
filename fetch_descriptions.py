import pandas as pd
from imdb import Cinemagoer
from tqdm import tqdm

links = pd.read_csv('data/ml-20m/links.csv', header=0, names=["movie_id", "imdb_id", "tmdb_id"])

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

descriptions = links[["movie_id", "description"]]

descriptions.to_csv("data/ml-20m/descriptions.csv", index=False)