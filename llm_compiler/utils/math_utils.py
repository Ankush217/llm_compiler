"""
Math Utilities
==============

Mathematical utilities for constraint solving and dimension calculations.
"""

import math
from typing import List, Tuple, Optional

def round_to_multiple(x: int, multiple: int) -> int:
    """Round integer to nearest multiple"""
    return ((x + multiple // 2) // multiple) * multiple

def find_divisors(n: int, min_val: int = 1, max_val: int = None) -> List[int]:
    """Find divisors of n within range"""
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            if min_val <= i <= (max_val or n):
                divisors.append(i)
            if i != n // i and min_val <= n // i <= (max_val or n):
                divisors.append(n // i)
    return sorted(divisors)

def find_closest_divisor(n: int, target: int) -> int:
    """Find divisor of n closest to target"""
    divisors = find_divisors(n)
    if not divisors:
        return 1
    return min(divisors, key=lambda x: abs(x - target))

def solve_quadratic(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float]]:
    """Solve quadratic equation"""
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None, None
    sqrt_disc = math.sqrt(discriminant)
    x1 = (-b + sqrt_disc) / (2*a)
    x2 = (-b - sqrt_disc) / (2*a)
    return x1, x2

def estimate_transformer_params(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_heads: int,
    num_kv_heads: int,
    tie_embeddings: bool = True
) -> int:
    """Estimate transformer parameters"""
    # Embeddings
    params = vocab_size * hidden_size
    
    # Per layer
    per_layer = 0
    
    # Attention
    head_dim = hidden_size // num_heads
    q_size = num_heads * head_dim
    k_size = num_kv_heads * head_dim
    v_size = num_kv_heads * head_dim
    
    per_layer += hidden_size * (q_size + k_size + v_size)  # QKV
    per_layer += (num_heads * head_dim) * hidden_size  # Output
    
    # MLP (SwiGLU)
    per_layer += 2 * hidden_size * intermediate_size  # gate/up
    per_layer += intermediate_size * hidden_size  # down
    
    # Norms (RMSNorm - scale only)
    per_layer += 2 * hidden_size
    
    # Total layers
    params += per_layer * num_layers
    
    # Output layer (if not tied)
    if not tie_embeddings:
        params += hidden_size * vocab_size
    
    return params

def solve_for_hidden_size(
    target_params: int,
    num_layers: int,
    vocab_size: int,
    intermediate_multiplier: float = 2.6875,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    tie_embeddings: bool = True
) -> int:
    """
    Solve for hidden size given target parameters.
    
    Based on transformer parameter formula:
    params = vocab * hidden + num_layers * (
        hidden * (hidden * (3 + 1)) +  # QKV + output
        2 * hidden * intermediate +    # MLP gate/up
        intermediate * hidden +        # MLP down
        2 * hidden                     # Norms
    )
    + (not tied) * hidden * vocab
    """
    # Intermediate size
    def intermediate(hidden):
        return int(round(hidden * intermediate_multiplier / 32) * 32)
    
    # Solve quadratic equation
    # Derived from parameter formula
    head_dim = 128  # assumption
    
    # Coefficients
    a = num_layers * (
        4 +  # QKV + output
        3 * intermediate_multiplier  # MLP
    )
    
    if not tie_embeddings:
        a += vocab_size
    
    b = 2 * num_layers  # Norms
    c = -target_params
    
    # Solve
    hidden = (-b + math.sqrt(b**2 - 4*a*c)) / (2*a)
    
    return int(round(hidden))

def validate_dimensions(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int
) -> List[str]:
    """Validate dimension constraints"""
    errors = []
    
    if hidden_size % num_heads != 0:
        errors.append(f"hidden_size ({hidden_size}) not divisible by num_heads ({num_heads})")
    
    if hidden_size % num_kv_heads != 0:
        errors.append(f"hidden_size ({hidden_size}) not divisible by num_kv_heads ({num_kv_heads})")
    
    if num_heads % num_kv_heads != 0:
        errors.append(f"num_heads ({num_heads}) not divisible by num_kv_heads ({num_kv_heads})")
    
    return errors