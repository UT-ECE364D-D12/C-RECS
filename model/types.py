from typing import List, Tuple

from torch import Tensor

ContentFeatures = Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tuple[List[str], Tensor], Tuple[List[str], Tensor]]

CollaborativeFeatures = Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tuple[List[str], Tensor], Tensor]

# Embedding, Logits, IDs
Anchor = Tuple[Tensor, Tensor, Tensor]
Positive = Tuple[Tensor, Tensor, Tensor]
Negative = Tuple[Tensor, Tensor, Tensor]
