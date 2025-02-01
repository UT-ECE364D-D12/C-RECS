from typing import Dict, List, Tuple

import torch
from torch import Tensor, cosine_similarity, nn

from model.encoder import Encoder
from model.layers import MultiLayerPerceptron
from model.recommender import DeepFM
from utils.misc import take_annotation_from


class CRECS(nn.Module):
    """
    Conversational Recommender System.

    Args:
        classifier (MultiLayerPerceptron): Classifier to predict items IDs, optional.
        weights (str): Path to the model weights, optional.
        **kwargs: Additional arguments for the Encoder, Recommender, and Classifier.
    """

    def __init__(self, classifier: MultiLayerPerceptron = None, weights: str = None, **kwargs) -> None:
        super().__init__()

        self.recommender = DeepFM(**kwargs["recommender"])

        self.encoder = Encoder(**kwargs["encoder"])

        self.classifier = classifier

        if weights is not None:
            self.load_weights(weights)

    def forward(
        self,
        rec_features: Tuple[List[Tensor], List[Tensor], Tensor],
        anchor_requests: List[str],
        anchor_ids: Tensor,
        negative_ids: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Forward pass of the model used during training.

        Args:
            rec_features (Tuple[List[Tensor], List[Tensor], Tensor]): User and item features.
            anchor_requests (List[str]): User requests.
            anchor_ids (Tensor): Positive item IDs.
            negative_ids (Tensor): Negative item IDs.

        Returns:
            rec_predictions (Tensor): Predicted ratings for the positive items.
            anchor (Tuple[Tensor, Tensor, Tensor]): Anchor request embeddings, logits, and IDs.
            positive (Tuple[Tensor, Tensor, Tensor]): Positive item embeddings, logits, and IDs.
            negative (Tuple[Tensor, Tensor, Tensor]): Negative item embeddings, logits, and IDs.
        """

        assert self.classifier is not None, "Classifier must be defined during training."

        rec_predictions = self.recommender(rec_features)

        # Get request/item embeddings: Anchor (Request), Positive (Item), Negative (Random Item)
        anchor_embeddings = self.encoder(anchor_requests)
        positive_embeddings = self.recommender.embedding.item_embedding(anchor_ids)
        negative_embeddings = self.recommender.embedding.item_embedding(negative_ids)

        # Predict the anchor, positive, and negative ids
        anchor_logits = self.classifier(anchor_embeddings)
        positive_logits = self.classifier(positive_embeddings)
        negative_logits = self.classifier(negative_embeddings)

        # Return the predictions and the triplet pairs
        anchor = (anchor_embeddings, anchor_logits, anchor_ids)
        positive = (positive_embeddings, positive_logits, anchor_ids)
        negative = (negative_embeddings, negative_logits, negative_ids)

        return rec_predictions, anchor, positive, negative

    def predict(self, rec_features: Tuple[Tensor, Tensor], request: str, k: int = 10) -> Tuple[Tensor, Tensor]:
        """
        Predict how the user would rate the k items closest to the user request.

        Args:
            rec_features (Tuple[Tensor, Tensor]): User features to predict the ratings.
            request (str): User request.
            k (int): Number of items to recommend.

        Returns:
            item_ids (Tensor): IDs of the k items, sorted by relevance.
            item_ratings (Tensor): Predicted ratings for the k items.
        """

        # Predict the ratings for all items based on the user features
        ratings = self.recommender.predict(rec_features)

        # Find the k most relevant items based on the user request
        request_embeddings = self.encoder(request)

        similarities = cosine_similarity(request_embeddings, self.recommender.embedding.item_embedding.weight)

        _, item_ids = torch.topk(similarities, k=k, largest=True)

        item_ratings = ratings[item_ids]

        # Sort the item ids based on the predicted ratings
        _, ranking = torch.sort(item_ratings, descending=True)

        return item_ids[ranking], item_ratings[ranking]

    def load_weights(self, path: str) -> None:
        """
        Load the model weights from a file.

        Args:
            path (str): Path to the model weights.
        """

        state_dict: Dict = torch.load(path, map_location="cpu", weights_only=True)

        if self.classifier is None:
            state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}

        self.load_state_dict(state_dict)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
