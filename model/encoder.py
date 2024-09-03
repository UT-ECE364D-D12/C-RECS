import logging
from typing import List

import torch
from torch import Tensor, nn
from transformers import BertModel, BertTokenizer

from model.layers import MultiLayerPerceptron


class Encoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            if "transformers" in logger.name.lower():
                logger.setLevel(logging.ERROR)
                
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model: BertModel = BertModel.from_pretrained("bert-base-uncased", **kwargs)
    
    def forward(self, requests: List[str]) -> Tensor:
        encoder_tokens = self.tokenizer(requests, padding=True, return_tensors="pt").to(self.model.device)

        batch_encoded_requests = self.model(**encoder_tokens)

        encoded_requests = batch_encoded_requests.last_hidden_state[:, 0]

        return encoded_requests

def build_expander(embed_dim: int, width: float = 2.0, **kwargs) -> MultiLayerPerceptron:
    return MultiLayerPerceptron(input_dim=embed_dim, hidden_dims=[expander_dim := int(embed_dim * width), expander_dim], output_dim=expander_dim, **kwargs)

