import torch
import logging
import time
import numpy as np


"""
Remember to train with HuggingFace's accelerate 
"""

class PositionalEncoding:
    def __init__(self) -> None:
        pass


class ScaledDotProductAttention:
    def __init__(
        self,
        embedding_size: int,
        query: np.ndarray,
        keys: np.ndarray,
        value: np.ndarray,
    ) -> None:
        self.query = query
        self.keys = keys
        self.value = value


class MultiHeadAttention(torch.nn.Module):
    """
    Each attention head has its set of 3 linear projection matrices,
    namely W_q, W_k, and W_v.
    """

    def __init__(self, num_heads: int = 8) -> None:
        super().__init__()
        self.attention_heads = num_heads


class Encoder(torch.nn.Module):
    def __init__(self, num_layers=6) -> None:
        super().__init__()
        self.num_layers = num_layers

    def forwad(self, x):
        pass


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
