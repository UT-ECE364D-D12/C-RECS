import os
import sys

# Add your base directory to the Python path
BASE_DIR = '/home/haakon/programs/senior-design/C-RECS'
sys.path.append(BASE_DIR)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
import yaml
from torch import cosine_similarity

from model.encoder import Encoder
from model.recommender import DeepFM
from utils.misc import send_to_device

app = Flask(__name__)
CORS(app)

# Load all resources and models once when the app starts
movies = pd.read_csv("data/ml-20m/movies.csv")
movies = movies[["item_id", "item_title"]]
num_items = movies["item_id"].nunique()

# Load configuration and initialize models
args = yaml.safe_load(open("configs/collaborative.yaml", "r"))
args["recommender"]["weights"] = "weights/deepfm.pt"
args["encoder"]["weights"] = "weights/encoder.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models and load weights
encoder = Encoder(**args["encoder"]).to(device)
encoder.load_state_dict(torch.load(args["encoder"]["weights"], map_location=device))

recommender = DeepFM(num_items=num_items, **args["recommender"]).to(device)
recommender.load_state_dict(torch.load(args["recommender"]["weights"], map_location=device))
recommender.eval()  # Set model to evaluation mode

item_embeddings = recommender.embedding.item_embedding.weight.cpu()

@app.route('/api/chat', methods=['POST'])
def chat():
    print("Received a request")

    data = request.get_json()
    user_message = data.get('message', '')

    # Encode the user message to get the request embedding
    request_embedding = encoder(user_message).cpu()

    # Calculate similarity between the request and item embeddings
    similarities = (cosine_similarity(request_embedding, item_embeddings) + 1) / 2

    # Dummy user features for the recommendation (can be modified as needed)
    features = (torch.tensor([num_items, 0]), torch.tensor([5.0, 3.5]))
    features = send_to_device(features, device)

    # Predict user ratings
    user_ratings = recommender.predict(features).cpu()

    # Compute rankings based on ratings and similarity
    # Get the top k similarities
    top_k_similarities, top_k_indices = torch.topk(similarities, k=20, largest=True)

    # Rank the top k items based on user ratings
    top_k_ratings = user_ratings[top_k_indices]
    _, ranking_indices = torch.sort(top_k_ratings, descending=True)

    # Get the final rankings
    rankings = top_k_indices[ranking_indices]

    # Retrieve the top recommended movies
    recommended_movies = movies.iloc[rankings[:5]].item_title.tolist()

    # Create response message
    response_message = f"Here are some movie recommendations for you: {', '.join(recommended_movies)}"


    print(f"User message: {user_message}")
    print(f"Response message: {response_message}")

    return jsonify({"response": response_message})

if __name__ == '__main__':
    app.run(debug=True)
