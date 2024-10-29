from typing import List, Tuple

import torch

from model.encoder import Encoder
from model.recommender import DeepFM
from benchmarking.metrics import ndcg_at_k, hr_at_k



