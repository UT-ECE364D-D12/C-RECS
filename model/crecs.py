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

        self.mlp = MultiLayerPerceptron(input_dim=self.encoder.embed_dim * 3, **kwargs["mlp"])

        self.classifier = classifier

    def forward(
        self, 
        rec_features: Tuple[List[Tensor], List[Tensor], Tensor],
        anchor_requests: str,
        anchor_ids: Tensor,
        negative_ids: Tensor
        ) -> None:

        assert self.classifier is not None, "Classifier must be defined during training."

        rec_predictions = self.recommender(rec_features)

        # Get request/item embeddings: Anchor (Request), Positive (Item), Negative (Random Item)
        anchor_embeddings = self.encoder(anchor_requests)
        negative_embeddings = self.recommender.embedding.item_embedding(negative_ids)

        embeddings = self.recommender.embedding(rec_features)
        user_embeddings, positive_embeddings = embeddings[:, 0], embeddings[:, 1]

        # Predict the similarity between the request and the positive/negative items
        ap_similarity = sigmoid(self.mlp(torch.cat((user_embeddings, anchor_embeddings, positive_embeddings), dim=1)))
        an_similarity = sigmoid(self.mlp(torch.cat((user_embeddings, anchor_embeddings, negative_embeddings), dim=1)))

        # Predict the anchor, positive, and negative ids
        anchor_logits = self.classifier(anchor_embeddings)
        positive_logits = self.classifier(positive_embeddings)
        negative_logits = self.classifier(negative_embeddings)

        return rec_predictions, (anchor_embeddings, anchor_logits, anchor_ids), (positive_embeddings, positive_logits, anchor_ids), (negative_embeddings, negative_logits, negative_ids), (ap_similarity, an_similarity)



    def predict(self):
        raise NotImplementedError
