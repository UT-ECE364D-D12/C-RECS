# C-RECS

Conversational Recommender System
## Setup

1. Create an environment:
```bash
mamba env create -f environment.yaml
```

2. Activate the environment:
```bash
conda activate crecs 
```

3. Compile bitsandbytes from source:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```

4. Download MovieLens 20M:
```bash
bash download_data.sh
```

5. Preprocess the data:
```bash
python preprocess.py
```

## Training

1. Ensure you have access to all of the LLM's used in `simulate_requests.py`, and that you are logged into huggingface:
```bash
huggingface-cli login
```

2. Log in to W&B to view run metrics:
```bash
wandb login
```

To jointly train the encoder and recommender using collaborative filtering:

1. Simulate item requests:
```bash
python simulate_requests.py
```

2. Train the encoder & recommender:
```bash
python train_collaborative.py
```

To train the encoder using content filtering:

1. Generate item descriptions:
```bash
python generate_descriptions.py
```

2. Train the encoder:
```bash
python train_content.py
```

# Dataset
This repository uses [MovieLens](https://grouplens.org/datasets/movielens/), a dataset of movie ratings.
