from utils.misc import send_to_device
import logging
from typing import List

import torch
from torch import Tensor, nn

from model.layers import MultiLayerPerceptron
from sentence_transformers import SentenceTransformer

class Encoder(nn.Module):
    def __init__(self, model_name: str = "all-mpnet-base-v2", weights: str = None, **kwargs) -> None:
        super().__init__()
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            if "transformers" in logger.name.lower():
                logger.setLevel(logging.ERROR)
                
        self.model = SentenceTransformer(model_name, config_kwargs=kwargs)

        self.embed_dim = self.model.get_sentence_embedding_dimension()
        
        if weights is not None:
            self.load_state_dict(torch.load(weights, weights_only=True))
    
    def forward(self, requests: List[str]) -> Tensor:
        
        request_tokens = send_to_device(self.model.tokenize(requests), device=self.model.device)

        encoded_requests = self.model(request_tokens)["sentence_embedding"]

        return encoded_requests
    
    def __call__(self, *args) -> Tensor:
        return super().__call__(*args)

def build_expander(embed_dim: int, width: float = 2.0, **kwargs) -> MultiLayerPerceptron:
    return MultiLayerPerceptron(input_dim=embed_dim, hidden_dims=[expander_dim := int(embed_dim * width), expander_dim], output_dim=expander_dim, **kwargs)

def build_classifier(embed_dim: int, num_classes: int, **kwargs) -> MultiLayerPerceptron:
    return MultiLayerPerceptron(input_dim=embed_dim, hidden_dims=[embed_dim, embed_dim], output_dim=num_classes, **kwargs)
