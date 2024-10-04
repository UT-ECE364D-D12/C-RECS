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

4. Download MovieLens and place it in `data/ml-20m/`:
```bash
bash fetch_data.bsh
```

5. Preprocess the data
```bash
python preprocess.py
```

## Training

To jointly train the encoder and recommender:

1. Ensure you have access to all of the LLM's used in `simulate_requests.py`, and that you are logged into huggingface:
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
