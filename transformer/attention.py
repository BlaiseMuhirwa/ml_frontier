import torch
import logging
import time
import math
import numpy as np


"""
Remember to train with HuggingFace's accelerate 
"""


class PositionalEncoding(torch.nn.Module):
    """
    This layer represents the positional encoding (embedding), which is summed up
    with the embedding for each word in the original input sequence.
    Args:
        - `dimension`: Represents the dimension of each word embedding.
        - `dropout_rate`: The percentage of neurons to randomly dropout
        - `max_sequence_length`: The maximum sequence length for a given input
            in the batch

    Returns: Sums up the word embedding with the computed positional encodings

    Note: Notice that the positional encodings are not learned parameters in the
        original transformers paper
    """

    def __init__(
        self, dimension: int, dropout_rate: float = 0.1, max_sequence_length: int = 5000
    ) -> None:
        self.dropout = torch.nn.Dropount(d=dropout_rate)
        positional_encoding = torch.zeros(
            (max_sequence_length, 1, dimension), dtype=torch.float32
        )
        multiplicative_factor = -math.log(10000) / dimension
        for position in range(max_sequence_length):
            for dim in range(0, dimension, 2):
                denominator = torch.exp(position * multiplicative_factor)
                positional_encoding[position, 0, dim] = torch.sin(
                    position * denominator
                )
                positional_encoding[position, 0, dim + 1] = torch.cos(
                    position * denominator
                )

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: input tensor of shape (max_sequence_length, batch_size, dimension)
        """
        input = input + self.positional_encoding[:input.shape[0]]
        input = self.dropout(input)
        return input


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

    def __init__(self, dimension: int, num_heads: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.attention_heads = num_heads

    def forward(self, x):
        pass


class EncoderBlock(torch.nn.Module):
    def __init__(
        self, dimension: int = 512, heads: int = 8, linear_layer_dim: int = 1024
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.heads = heads
        self.linear = torch.nn.Linear(dimension, linear_layer_dim)
        self.multi_head_attention = MultiHeadAttention(
            dimension=dimension, num_heads=heads
        )

    def forward(self, x):
        pass


class DecoderBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


class Encoder(torch.nn.Module):
    def __init__(self, num_blocks: int = 6) -> None:
        super().__init__()
        self.num_blocks = num_blocks

    def forward(self, x):
        pass


class Decoder(torch.nn.Module):
    def __init__(self, num_blocks: int = 6) -> None:
        super().__init__()
        self.num_blocks = num_blocks

    def forward(self, x):
        pass
