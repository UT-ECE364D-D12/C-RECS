import logging
from typing import Tuple

from transformers import BertModel, BertTokenizer

from model.layers import MultiLayerPerceptron


def build_encoder(device: str = "cpu") -> Tuple[BertModel, BertTokenizer]:
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.ERROR)
            
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)

    return model, tokenizer

def build_expander(embed_dim: int, width: float = 2.0) -> MultiLayerPerceptron:
    expander_dim = int(embed_dim * width)

    return MultiLayerPerceptron(input_dim=embed_dim, hidden_dims=[expander_dim, expander_dim], output_dim=expander_dim, dropout=0.0)

