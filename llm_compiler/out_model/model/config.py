
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32
    hidden_size: int = 16
    intermediate_size: int = 11008
    num_hidden_layers: int = 1
    num_attention_heads: int = 2
    num_key_value_heads: int = 1
    head_dim: int = 8
    max_position_embeddings: int = 16

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    torch_dtype: str = "float32"

    def to_dict(self):
        return self.__dict__.copy()
