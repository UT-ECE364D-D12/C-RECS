import os
import sys

# Add your base directory to the Python path
BASE_DIR = './'
sys.path.append(BASE_DIR)

from typing import Dict

import pandas as pd
import torch
import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS

from model.crecs import CRECS
from utils.misc import send_to_device

app = Flask(__name__)
CORS(app)

# Load the items 
items = pd.read_csv("data/ml-20m/movies.csv")[["item_id", "item_title"]]

num_items = items["item_id"].nunique()

# Initialize models and load weights
args = yaml.safe_load(open("configs/collaborative.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

args["model"]["recommender"]["num_items"] = num_items 

model = CRECS(weights="weights/collaborative/crecs.pt", **args["model"]).to(device)

@app.route('/api/chat', methods=['POST'])
def chat():
    data: Dict = request.get_json()
    user_message = data.get('message', '')

    print(f"Received request: {user_message}")

    # TODO: Hardcoded user features
    features = (torch.tensor([num_items]), torch.tensor([5.0]))

    features = send_to_device(features, device)

    item_ids, item_ratings = send_to_device(model.predict(features, user_message, k=10), "cpu")
    
    recommended_items = list(items.iloc[item_ids]["item_title"])[:5]

    # Create response message
    response_message = f"Here are some movie recommendations for you: {', '.join(recommended_items)}"

    return jsonify({"response": response_message})

if __name__ == '__main__':
    app.run(debug=True)
