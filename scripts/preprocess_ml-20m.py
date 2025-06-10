# Preprocess the MovieLens 20M dataset
import argparse
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def preprocess(
    data_root: Path,
    min_ratings_per_movie: int = 20,
    min_ratings_per_user: int = 10,
    num_ratings_to_sample: int = 2000000,
    train_size: float = 0.8,
    val_size: float = 0.1,
) -> None:
    """
    Preprocess the MovieLens 20M dataset.

    Removes duplicate movies, filters out movies and users with insufficient ratings,
    samples a subset of users, and converts the ratings into a format suitable for training.

    Args:
        data_root (Path): The root directory where the dataset is stored.
        min_ratings_per_movie (int): Minimum number of ratings a movie must have to be included.
        min_ratings_per_user (int): Minimum number of ratings a user must have to be included.
        num_ratings_to_sample (int): Number of ratings to sample from the dataset.
        train_size (float): Proportion of the users to include in the train split.
        val_size (float): Proportion of the users to include in the validation split.
    """

    # Load movies and ratings data
    movies = pd.read_csv(
        data_root / "movies.csv",
        header=0,
        names=["item_id", "item_title", "genres"],
        dtype={"item_id": int, "item_title": str, "genres": str},
    )

    ratings = pd.read_csv(
        data_root / "ratings.csv",
        header=0,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )

    # Remove duplicate movies
    ratings = pd.merge(ratings, movies[["item_id", "item_title"]], on="item_id")
    ratings = ratings[["user_id", "item_title", "rating", "timestamp"]]
    movies = movies.drop_duplicates("item_title")
    ratings = pd.merge(ratings, movies, on="item_title")
    ratings = ratings[["user_id", "item_id", "rating", "timestamp"]]

    # Remove movies that have no ratings
    movies = movies[movies["item_id"].isin(ratings["item_id"])]

    print(f"Loaded movies and ratings data from {data_root}")
    print(f"{len(movies):,} movies, {ratings['user_id'].nunique():,} users, {len(ratings):,} ratings\n")

    # Sample `num_ratings_to_sample` ratings
    ratings = ratings.sample(n=num_ratings_to_sample, random_state=42, replace=False).reset_index(drop=True)
    movies = movies[movies["item_id"].isin(ratings["item_id"])]

    print(f"Sampled {num_ratings_to_sample} ratings")
    print(f"{len(movies):,} movies, {ratings['user_id'].nunique():,} users, {len(ratings):,} ratings\n")

    # Remove movies with less than `min_ratings_per_movie` ratings
    num_movie_ratings = ratings.groupby("item_id").size()

    ratings = ratings[ratings["item_id"].isin(num_movie_ratings[num_movie_ratings >= min_ratings_per_movie].index)]
    movies = movies[movies["item_id"].isin(ratings["item_id"])]

    print(f"Removed movies with less than {min_ratings_per_movie} ratings")
    print(f"{len(movies):,} movies, {ratings['user_id'].nunique():,} users, {len(ratings):,} ratings\n")

    # Remove users with less than `min_ratings_per_user` ratings
    num_user_ratings = ratings.groupby("user_id").size()

    ratings = ratings[ratings["user_id"].isin(num_user_ratings[num_user_ratings >= min_ratings_per_user].index)]
    movies = movies[movies["item_id"].isin(ratings["item_id"])]

    print(f"Removed users with less than {min_ratings_per_user} ratings")
    print(f"{len(movies):,} movies, {ratings['user_id'].nunique():,} users, {len(ratings):,} ratings\n")

    # Convert each user_id and item_id to a sequential unique_id
    user_id_to_unique_id = {user_id: i for i, user_id in enumerate(ratings["user_id"].unique())}
    item_id_to_unique_id = {item_id: i for i, item_id in enumerate(movies["item_id"].unique())}

    movies["item_id"] = movies["item_id"].map(item_id_to_unique_id)

    ratings["user_id"] = ratings["user_id"].map(user_id_to_unique_id)
    ratings["item_id"] = ratings["item_id"].map(item_id_to_unique_id)

    # Split into training and test sets by users
    test_size = 1 - train_size - val_size
    val_size = val_size / (train_size + val_size)
    train_users, test_users = train_test_split(ratings["user_id"].unique(), test_size=test_size, random_state=42)
    train_users, val_users = train_test_split(train_users, test_size=val_size, random_state=42)

    train_ratings = ratings[ratings["user_id"].isin(train_users)].reset_index(drop=True)
    val_ratings = ratings[ratings["user_id"].isin(val_users)].reset_index(drop=True)
    test_ratings = ratings[ratings["user_id"].isin(test_users)].reset_index(drop=True)

    # Remove items that are not in the training set
    movies = movies[movies["item_id"].isin(train_ratings["item_id"].unique())]
    ratings = ratings[ratings["item_id"].isin(movies["item_id"])]

    train_ratings = train_ratings[train_ratings["item_id"].isin(movies["item_id"])]
    val_ratings = val_ratings[val_ratings["item_id"].isin(movies["item_id"])]
    test_ratings = test_ratings[test_ratings["item_id"].isin(movies["item_id"])]

    no_item_id = ratings["item_id"].nunique()

    print(f"Split into train, validation, and test sets")
    print(f"{len(movies):,} movies, {ratings['user_id'].nunique():,} users, {len(ratings):,} ratings\n")

    def process_ratings(ratings: DataFrame) -> DataFrame:
        # Process ratings by converting them into a format suitable for training
        grouped_user_ratings = ratings.sort_values(by=["user_id", "timestamp"]).groupby("user_id")
        grouped_user_ratings = grouped_user_ratings.agg({"item_id": list, "rating": list, "timestamp": list}).reset_index()

        # Each training example will consist of the items the user has rated so far, along with the new item they are rating.
        processed_ratings = []

        for user_id, item_ids, item_ratings, timestamps in grouped_user_ratings.values:
            feature_ids = [no_item_id]
            feature_ratings = [5.0]

            for item_id, rating, timestamp in zip(item_ids, item_ratings, timestamps):
                processed_ratings.append(
                    {
                        "user_id": user_id,
                        "feature_ids": feature_ids.copy(),
                        "feature_ratings": feature_ratings.copy(),
                        "item_id": item_id,
                        "rating": rating,
                        "timestamp": timestamp,
                    }
                )
                feature_ids.append(item_id)
                feature_ratings.append(rating)

        return DataFrame(processed_ratings)

    print("Processing train set...")
    train_ratings = process_ratings(train_ratings)

    print("Processing validation set...")
    val_ratings = process_ratings(val_ratings)

    print("Processing test set...")
    test_ratings = process_ratings(test_ratings)

    # Save the movies and ratings data
    movies.to_csv(data_root / "movies.csv", index=False)
    ratings.to_csv(data_root / "ratings.csv", index=False)

    # Save the processed ratings data
    train_ratings.to_parquet(data_root / "train_ratings.parquet", index=False, engine="pyarrow", compression="snappy")
    val_ratings.to_parquet(data_root / "val_ratings.parquet", index=False, engine="pyarrow", compression="snappy")
    test_ratings.to_parquet(data_root / "test_ratings.parquet", index=False, engine="pyarrow", compression="snappy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, help="Root directory of MovieLens data")
    args = parser.parse_args()

    preprocess(args.data_root)
