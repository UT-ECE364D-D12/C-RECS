import numpy as np
import pandas as pd

def ndcg_at_k(rating_true: pd.DataFrame, rating_pred: pd.DataFrame, k: int = 10) -> float:
    def dcg(scores):
        return sum([score / np.log2(idx + 2) for idx, score in enumerate(scores)])

    total_ndcg = 0.0
    user_count = rating_true['user_id'].nunique()

    for user in rating_true['user_id'].unique():
        # Get true and predicted ratings for this user
        true_ratings = rating_true[rating_true['user_id'] == user].nlargest(k, 'rating')['rating'].tolist()
        pred_ratings = rating_pred[rating_pred['user_id'] == user].nlargest(k, 'predicted')['predicted'].tolist()

        # Calculate DCG for predictions and ideal DCG
        dcg_pred = dcg(pred_ratings)
        dcg_ideal = dcg(true_ratings)

        total_ndcg += (dcg_pred / dcg_ideal) if dcg_ideal > 0 else 0.0

    return total_ndcg / user_count if user_count > 0 else 0.0


def hr_at_k(rating_true: pd.DataFrame, rating_pred: pd.DataFrame, k: int = 10) -> float:
    hit_count = 0
    user_count = rating_true['user_id'].nunique()

    for user in rating_true['user_id'].unique():
        true_items = set(rating_true[rating_true['user_id'] == user].nlargest(k, 'rating')['item_id'].tolist())
        pred_items = set(rating_pred[rating_pred['user_id'] == user].nlargest(k, 'predicted')['item_id'].tolist())

        # Check for hits
        if true_items.intersection(pred_items):
            hit_count += 1

    return hit_count / user_count if user_count > 0 else 0.0


def evaluate_with_request_context(user_id, request, items, true_items, deepfm_model, text_encoder):
    """
    Evaluate a recommendation model only against items that match the request context.
    """

    # temp stuff to remove errors (import functions later)
    def cosine_similarity(a, b):
        return 0.0
    def get_item_embedding(item):
        return np.zeros(0)
    def get_hybrid_recommendations(user_id, request, items, deepfm_model, text_encoder):
        return []


    # Filter ground truth items to only those matching the request context
    request_embedding = text_encoder(request)
    
    # Define a threshold for what's considered "matching" the request
    SIMILARITY_THRESHOLD = 0.7
    
    # Filter ground truth to only items matching the request context
    contextual_ground_truth = [
        item for item in true_items 
        if cosine_similarity(request_embedding, get_item_embedding(item)) > SIMILARITY_THRESHOLD
    ]
    
    # Now evaluate only against these contextually relevant items
    hybrid_recs = get_hybrid_recommendations(user_id, request, items, deepfm_model, text_encoder)
    hit10 = any(item in contextual_ground_truth for item, _ in hybrid_recs[:10])
    
    return hit10
