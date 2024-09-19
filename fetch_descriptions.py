import pandas as pd
from imdb import Cinemagoer
from tqdm import tqdm

links = pd.read_csv('data/ml-20m/links.csv', header=0, names=["movie_id", "imdb_id", "tmdb_id"])

links = links.head()

imdb = Cinemagoer()

def get_description(imdb_id: str):
    try:
        return imdb.get_movie(imdb_id)["plot outline"]
    except:
        return None
    
tqdm.pandas(desc="Fetching Descriptions", unit="movie")

links["description"] = links["imdb_id"].progress_apply(get_description)

descriptions = links[["movie_id", "description"]]

descriptions.to_csv("data/ml-20m/descriptions.csv", index=False)