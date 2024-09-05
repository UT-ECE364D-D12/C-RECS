# CERS
Chatbot Enhanced Recommender System

## Setup

1. Create an environment:
```bash
conda env create -f environment.yaml
```
Or
```bash
conda env create -f cpu_environment.yaml
```
2. Compile bitsandbytes from source (If GPU):
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install -e .
```

3. Download [MovieLens](https://grouplens.org/datasets/movielens/100k/) and place it in `data/ml-100k/`.

# Dataset
This repository uses [MovieLens](https://grouplens.org/datasets/movielens/), a dataset that contains 25M ratings for 160k users across 62k movies.
