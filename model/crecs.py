from typing import List, Tuple

import torch
from torch import Tensor, nn, sigmoid

from model.encoder import Encoder
from model.layers import MultiLayerPerceptron
from model.recommender import DeepFM


class CRECS(nn.Module):
    def __init__(self, classifier: MultiLayerPerceptron = None, **kwargs):
        super().__init__()

        self.recommender = DeepFM(**kwargs["recommender"])

        self.encoder = Encoder(**kwargs["encoder"])

        self.classifier = classifier

    def forward(
        self, 
        rec_features: Tuple[List[Tensor], List[Tensor], Tensor],
        anchor_requests: List[str],
        anchor_ids: Tensor,
        negative_ids: Tensor
        ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Return the predicted ratings and triplets for the given features and requests.
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


    def predict(self):
        raise NotImplementedError
