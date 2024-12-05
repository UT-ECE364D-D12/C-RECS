from typing import List, Tuple

import torch
from torch import Tensor, cosine_similarity, nn

from model.encoder import Encoder
from model.layers import MultiLayerPerceptron
from model.recommender import DeepFM


class CRECS(nn.Module):
    def __init__(self, classifier: MultiLayerPerceptron = None, weights: str = None, **kwargs):
        super().__init__()

        self.recommender = DeepFM(**kwargs["recommender"])

        self.encoder = Encoder(**kwargs["encoder"])

        self.classifier = classifier

        if weights is not None:
            state_dict = torch.load(weights, map_location="cpu", weights_only=True)

            if self.classifier is None: state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}

            self.load_state_dict(state_dict)

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

    def predict(self, rec_features: Tuple[Tensor, Tensor], requests: List[str], k: int = 10) -> Tuple[Tensor, Tensor]:
        """
        Return the recommended items along with their predicted ratings for the given features and requests. 
        """

        ratings = self.recommender.predict(rec_features)

        request_embeddings = self.encoder(requests)

        similarities = cosine_similarity(request_embeddings, self.recommender.embedding.item_embedding.weight)

        _, item_ids = torch.topk(similarities, k=k, largest=True)

        return item_ids, ratings[item_ids]

