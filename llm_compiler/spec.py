"""
Specification Layer
===================

Defines the declarative interface for specifying LLM architectures.
All model definitions must be complete and explicit - no magic defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
from enum import Enum

class NormType(Enum):
    """Normalization types"""
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    SCALENORM = "scalenorm"
    NO_NORM = "no_norm"

class ActivationType(Enum):
    """Activation functions"""
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    SWIGLU = "swiglu"
    GELLU = "gelu"
    RELU_SQUARED = "relu_squared"

class AttentionType(Enum):
    """Attention variants"""
    MHA = "mha"  # Multi-head attention
    MQA = "mqa"  # Multi-query attention
    GQA = "gqa"  # Grouped-query attention
    FLASH = "flash"  # Flash attention compatible
    SLIDING_WINDOW = "sliding_window"
    GLOBAL_LOCAL = "global_local"

class PositionalEncodingType(Enum):
    """Positional encoding methods"""
    ROPE = "rope"
    ALIBI = "alibi"
    SINUSOIDAL = "sinusoidal"
    RELATIVE = "relative"
    NONE = "none"

class TokenizerType(Enum):
    """Tokenizer types"""
    UNIGRAM = "unigram"
    BPE = "bpe"
    SENTENCEPIECE = "sentencepiece"
    WORDPIECE = "wordpiece"
    CHAR = "char"

class WeightFormat(Enum):
    """Weight storage formats"""
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    NUMPY = "numpy"
    GGUF = "gguf"

class BackendTarget(Enum):
    """Backend compilation targets"""
    PYTORCH_TRAINING = "pytorch_training"
    PYTORCH_INFERENCE = "pytorch_inference"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    JAX = "jax"

@dataclass
class LLM:
    """
    Complete LLM specification.
    
    All fields must be explicitly set - no automatic defaults.
    The system will validate and solve for missing dimensions.
    """
    name: str
    # Template selection
    template: str
    
    # Core dimensions (required, non-default)
    vocab_size: int
    context_length: int

    # Optional convenience dimension overrides
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    
    # Size specification (choose one)
    target_params: Optional[int] = None
    explicit_dims: Optional[Dict[str, int]] = None
    target_tolerance: float = 0.1  # acceptable relative error on parameter target
    
    # Architecture choices
    attention: AttentionType = AttentionType.GQA
    norm: NormType = NormType.RMSNORM
    activation: ActivationType = ActivationType.SWIGLU
    positional_encoding: PositionalEncodingType = PositionalEncodingType.ROPE
    
    # Tokenizer
    tokenizer: TokenizerType = TokenizerType.UNIGRAM
    
    # Output format
    weight_format: WeightFormat = WeightFormat.SAFETENSORS
    backend: BackendTarget = BackendTarget.PYTORCH_TRAINING
    
    # Optional advanced settings
    rope_theta: float = 10000.0
    rope_scaling_factor: Optional[float] = None
    alibi_max_bias: float = 8.0
    sliding_window_size: Optional[int] = None
    tie_word_embeddings: bool = True
    tie_embeddings: Optional[bool] = None
    gradient_checkpointing: bool = False
    precision: str = "float32"
    
    # Quantization (optional)
    quantize: Optional[str] = None
    quantize_bits: int = 16
    
    # Validation flag
    _validated: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Validate basic constraints after initialization"""
        # Fold convenience fields into explicit_dims
        if self.explicit_dims is None:
            self.explicit_dims = {}
        for key, value in [
            ("num_layers", self.num_layers),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_heads", self.num_heads),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
        ]:
            if value is not None:
                self.explicit_dims[key] = value

        if self.target_params is None and self.explicit_dims is None:
            raise ValueError("Must specify either target_params or explicit_dims")
        
        if self.target_params is not None and self.explicit_dims is not None:
            raise ValueError("Cannot specify both target_params and explicit_dims")
        
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        
        # Validate attention window if specified
        if (self.attention == AttentionType.SLIDING_WINDOW and 
            self.sliding_window_size is None):
            raise ValueError("sliding_window_size must be specified for sliding window attention")
        
        if (self.sliding_window_size is not None and 
            self.sliding_window_size > self.context_length):
            raise ValueError(f"sliding_window_size ({self.sliding_window_size}) "
                           f"exceeds context_length ({self.context_length})")

        if self.tie_embeddings is not None:
            self.tie_word_embeddings = self.tie_embeddings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary for serialization"""
        data = {}
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
            if isinstance(value, Enum):
                data[field.name] = value.value
            else:
                data[field.name] = value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLM':
        """Create spec from dictionary"""
        # Convert string enums back to Enum instances
        for field_name, field_type in cls.__dataclass_fields__.items():
            if field_name in data and hasattr(field_type.type, '__origin__'):
                if field_type.type.__origin__ == Union:
                    # Handle Optional types
                    for possible_type in field_type.type.__args__:
                        if hasattr(possible_type, '_member_names_'):
                            # This is an Enum type
                            if isinstance(data[field_name], str):
                                data[field_name] = possible_type(data[field_name])
                            break
                elif hasattr(field_type.type, '_member_names_'):
                    # This is an Enum type
                    if isinstance(data[field_name], str):
                        data[field_name] = field_type.type(data[field_name])
        
        return cls(**data)
