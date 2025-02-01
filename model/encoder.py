from importlib.util import find_spec
from typing import List

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from model.layers import MultiLayerPerceptron
from utils.misc import take_annotation_from


class Encoder(nn.Module):
    """
    Sentence encoder using a pre-trained transformer model.

    Args:
        model_name (str): Pre-trained model name, optional.
        weights (str): Path to the model weights, optional.
        **kwargs: Additional arguments for the model.
    """

    def __init__(self, model_name: str = "bert-base-uncased", weights: str = None, **kwargs) -> None:
        super().__init__()

        attn_implementation = None if find_spec("flash_attn") is None else "flash_attention_2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32,
            attn_implementation=attn_implementation,
            **kwargs,
        )

        self.dtype = torch.bfloat16 if attn_implementation == "flash_attention_2" else self.model.dtype
        self.embed_dim = self.model.config.hidden_size

        if weights is not None:
            self.load_state_dict(torch.load(weights, weights_only=True))

    def forward(self, requests: List[str]) -> Tensor:
        """
        Encodes a list of requests into a tensor.

        Args:
            requests (List[str]): List of requests.

        Returns:
            encoded_requests (Tensor): Encoded requests of shape (batch_size, embed_dim).
        """

        device = self.model.device.__str__()

        with torch.autocast(device_type=device, dtype=self.dtype):
            encoder_tokens = self.tokenizer(requests, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

            batch_encoded_requests = self.model(**encoder_tokens)

            encoded_requests: Tensor = batch_encoded_requests.last_hidden_state[:, 0]

        return encoded_requests.to(torch.float32)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


def build_expander(embed_dim: int, width: float = 2.0, **kwargs) -> MultiLayerPerceptron:
    """
    Build a multi-layer perceptron expander, which expands the input dimensionality for VICReg.

    Args:
        embed_dim (int): Input embedding dimension.
        width (float): Width multiplier for the hidden layers.
        **kwargs: Additional arguments for the MultiLayerPerceptron.

    Returns:
        expander (MultiLayerPerceptron): Multi-layer perceptron expander.
    """

    return MultiLayerPerceptron(
        input_dim=embed_dim,
        hidden_dims=[expander_dim := int(embed_dim * width), expander_dim],
        output_dim=expander_dim,
        **kwargs,
    )


def build_classifier(embed_dim: int, num_classes: int, **kwargs) -> MultiLayerPerceptron:
    """
    Build a multi-layer perceptron classifier, used to predict the item IDs from the embeddings during training.

    Args:
        embed_dim (int): Input embedding dimension.
        num_classes (int): Number of classes.
        **kwargs: Additional arguments for the MultiLayerPerceptron.

    Returns:
        classifier (MultiLayerPerceptron): Multi-layer perceptron classifier.
    """

    return MultiLayerPerceptron(input_dim=embed_dim, hidden_dims=[embed_dim, embed_dim], output_dim=num_classes, **kwargs)
