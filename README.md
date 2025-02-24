# C-RECS

Conversational Recommender System

![Results](resources/crecs.jpeg)

## Install 

The steps to prepare the environment are outlined below.

### x86-64 Based Systems

1. Create an environment:
```bash
mamba env create -f environments/environment.yaml
```

2. Activate the environment:
```bash
conda activate crecs 
```

<!-- TODO: Remove after next transformers release includes answerdotai/ModernBERT-base -->
3. Install `transformers` from main:
```bash
pip install git+https://github.com/huggingface/transformers.git
```
> **_NOTE:_**  If you experience issues with `numpy`, try installing it separately using `mamba install -y "numpy==1.26.4"`

### aarch64 Based Systems

1. Create an environment:
```bash
mamba env create -f environments/tacc.yaml
```

2. Activate the environment:
```bash
conda activate crecs 
```

3. Install `torch`:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

<!-- TODO: Remove after next transformers release includes answerdotai/ModernBERT-base -->
4. Install `transformers` from main:
```bash
pip install git+https://github.com/huggingface/transformers.git
```

## Data

This repository uses [MovieLens](https://grouplens.org/datasets/movielens/), a dataset of movie ratings.

1. Download MovieLens 20M:
```bash
bash scripts/download_data.sh
```

2. Preprocess the data:
```bash
python preprocess.py
```

## Training

The steps to train the recommendation system are outlined below.

1. Ensure you have access to all of the LLM's used in `simulate_requests.py`, and that you are logged into huggingface:
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
python train_recommender.py
```

### Collaborative Filtering

To jointly train the encoder and recommender using collaborative filtering:

1. Update `configs/collaborative.yaml` file with the desired parameters.

2. Simulate item requests:
```bash
python simulate_requests.py
```

3. Train the encoder & recommender:
```bash
python train_collaborative.py
```

## App

1. Install the required packages for the app frontend:
```bash
cd app/frontend
npm install
``` 

2. Start the backend:
```bash
python app/backend/app.py
```

3. In a seperate terminal, start the frontend:
```bash
cd app/frontend
npm start
```
