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
2. (GPU Only) Compile bitsandbytes from source:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```

3. Download [MovieLens](https://grouplens.org/datasets/movielens/100k/) and place it in `data/ml-100k/`:
```bash
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
mkdir data && mv ml-100k.zip data/ && cd data/
unzip ml-100k.zip && rm ml-100k.zip
```

# Dataset
This repository uses [MovieLens](https://grouplens.org/datasets/movielens/), a dataset of movie ratings.
