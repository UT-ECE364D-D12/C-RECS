from importlib.util import find_spec
from typing import List

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from model.layers import MultiLayerPerceptron


class Encoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", weights: str = None, **kwargs) -> None:
        super().__init__()

        attn_implementation = None if find_spec("flash_attn") is None else "flash_attention_2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float32, attn_implementation=attn_implementation, **kwargs)

        self.dtype = torch.bfloat16 if attn_implementation == "flash_attention_2" else self.model.dtype 
        self.embed_dim = self.model.config.hidden_size

        if weights is not None:
            self.load_state_dict(torch.load(weights, weights_only=True))
    
    def forward(self, requests: List[str]) -> Tensor:
        with torch.autocast(device_type=self.model.device.__str__(), dtype=self.dtype):
            encoder_tokens = self.tokenizer(requests, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.model.device)

            batch_encoded_requests = self.model(**encoder_tokens)

            encoded_requests: Tensor = batch_encoded_requests.last_hidden_state[:, 0]

        return encoded_requests.to(torch.float32)
    
    def __call__(self, *args) -> Tensor:
        return super().__call__(*args)

def build_expander(embed_dim: int, width: float = 2.0, **kwargs) -> MultiLayerPerceptron:
    return MultiLayerPerceptron(input_dim=embed_dim, hidden_dims=[expander_dim := int(embed_dim * width), expander_dim], output_dim=expander_dim, **kwargs)

def build_classifier(embed_dim: int, num_classes: int, **kwargs) -> MultiLayerPerceptron:
    return MultiLayerPerceptron(input_dim=embed_dim, hidden_dims=[embed_dim, embed_dim], output_dim=num_classes, **kwargs)
