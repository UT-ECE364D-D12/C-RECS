# CERS
Chatbot Enhanced Recommender System

## Setup

1. Create an environment:
```bash
mamba env create -f environment.yaml
```
Or
```bash
mamba env create -f cpu_environment.yaml
```
2. Activate the environment:
```bash
conda activate cers
```

3. (GPU Only) Compile bitsandbytes from source:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```

4. Download [MovieLens](https://grouplens.org/datasets/movielens/100k/) and place it in `data/ml-100k/`:
```bash
mkdir data
cd data
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
rm ml-20m.zip
```


4. Ensure you have access to 

## Training

To jointly train the encoder and recommender:

1. Ensure you have access to all of the LLM's used in `simulate_requests.py`, and that you are logged into hugginface:
```bash
huggingface-cli login
```

2. Log in to W&B:
```bash
wandb login
```

3. Simulate movie requests:
```bash
python simulate_requests.py
```

4. Train the encoder & recommender:
```bash
python train_encoder.py
```

# Dataset
This repository uses [MovieLens](https://grouplens.org/datasets/movielens/), a dataset of movie ratings.
