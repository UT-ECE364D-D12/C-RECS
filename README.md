# C-RECS

Conversational Recommender System

![Results](resources/crecs.jpeg)

## Install 

The steps to prepare the environment are outlined below.

### x86-64 Based Systems

1. Create an environment:
```bash
conda env create -f environments/x86_64.yaml
```

2. Activate the environment:
```bash
conda activate crecs 
```

### aarch64 Based Systems

1. Create an environment:
```bash
conda env create -f environments/aarch64.yaml
```

2. Activate the environment:
```bash
conda activate crecs 
```

3. Build [bitsandbytes](https://huggingface.co/docs/bitsandbytes/en/installation) from source:
```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install  .
```

> **_NOTE:_**  If cmake does not recognize CUDA you may need to specify its location with `-DCMAKE_CUDA_COMPILER=<path_to_cuda_bin>`

## Data

This repository uses [MovieLens](https://grouplens.org/datasets/movielens/), a dataset of movie ratings.

1. Download MovieLens 20M:
```bash
./scripts/download_ml-20m.sh <data_root>
```

2. Preprocess the data:
```bash
python preprocess/preprocess_ml-20m.py <data_root>
```

## Training

The steps to train the recommendation system are outlined below.

1. Ensure you have access to all of the LLM's used in `simulate_ml-20m.py`, and that you are logged into huggingface:
```bash
huggingface-cli login
```

2. Log in to W&B to view run metrics:
```bash
wandb login
```

### Recommender

To train the recommender by itself:

1. Update `configs/recommender.yaml` file with the desired parameters.

2. Train the recommender:
```bash
python train.py --config configs/recommender.yaml
```

### Collaborative Filtering

To jointly train the encoder and recommender using collaborative filtering:

1. Update `configs/collaborative.yaml` file with the desired parameters.

2. Simulate item requests:
```bash
python scripts/simulate_ml-20m.py <data_root> --mode=requests
```

3. Train the encoder & recommender:
```bash
python train_collaborative.py
```