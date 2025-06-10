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

## Data

This repository uses [MovieLens](https://grouplens.org/datasets/movielens/), a dataset of movie ratings.

1. Download MovieLens 20M:
```bash
./scripts/download_ml-20m.sh <data_root>
```

2. Preprocess the data:
```bash
python scripts/preprocess_ml-20m.py <data_root>
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
python scripts/simulate_ml-20m.py
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
