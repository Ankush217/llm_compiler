"""
Decoder-Only Transformer Template
==================================

Implements GPT/LLaMA-style decoder-only architectures.
Supports MHA, MQA, GQA attention variants.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .base import ArchitectureTemplate, TemplateInfo, TemplateParameter
from ..ir.graph import IRGraph, Tensor, Operation, NodeType
from ..ir.builder import GraphBuilder
from ..solver.constraints import (
    Constraint, ConstraintSystem, EqualityConstraint, 
    DivisibilityConstraint, RangeConstraint, LinearConstraint
)
from ..utils.math_utils import round_to_multiple, find_divisors

class DecoderOnlyTemplate(ArchitectureTemplate):
    """Decoder-only transformer (GPT/LLaMA class)"""
    
    def __init__(self, version: str = "v1"):
        self.version = version
        
    @property
    def info(self) -> TemplateInfo:
        return TemplateInfo(
            name=f"decoder_only_{self.version}",
            description="Decoder-only transformer (GPT/LLaMA style)",
            version=self.version,
            parameters=[
                TemplateParameter.NUM_LAYERS,
                TemplateParameter.HIDDEN_SIZE,
                TemplateParameter.INTERMEDIATE_SIZE,
                TemplateParameter.NUM_HEADS,
                TemplateParameter.NUM_KV_HEADS,
                TemplateParameter.HEAD_DIM,
                TemplateParameter.VOCAB_SIZE,
                TemplateParameter.CONTEXT_LENGTH,
            ],
            required_parameters={
                TemplateParameter.VOCAB_SIZE,
                TemplateParameter.CONTEXT_LENGTH,
            },
            optional_parameters={
                TemplateParameter.NUM_LAYERS,
                TemplateParameter.HIDDEN_SIZE,
                TemplateParameter.INTERMEDIATE_SIZE,
                TemplateParameter.NUM_HEADS,
                TemplateParameter.NUM_KV_HEADS,
                TemplateParameter.HEAD_DIM,
            },
            default_constraints=[
                # Head dimension constraints
                DivisibilityConstraint(
                    var1=TemplateParameter.HIDDEN_SIZE.value,
                    var2=TemplateParameter.NUM_HEADS.value,
                    name="hidden_size_divisible_by_num_heads"
                ),
                DivisibilityConstraint(
                    var1=TemplateParameter.HIDDEN_SIZE.value,
                    var2=TemplateParameter.NUM_KV_HEADS.value,
                    name="hidden_size_divisible_by_num_kv_heads"
                ),
                # KV heads must divide query heads for GQA
                DivisibilityConstraint(
                    var1=TemplateParameter.NUM_HEADS.value,
                    var2=TemplateParameter.NUM_KV_HEADS.value,
                    name="num_heads_divisible_by_num_kv_heads"
                ),
                # Intermediate size typical expansion
                EqualityConstraint(
                    var1=TemplateParameter.INTERMEDIATE_SIZE.value,
                    var2=f"2.6875 * {TemplateParameter.HIDDEN_SIZE.value}",
                    name="swiglu_intermediate_size"
                ),
            ]
        )
    
    def create_constraint_system(self, spec: Dict[str, Any]) -> ConstraintSystem:
        """
        Create constraint system for decoder-only architecture.
        
        Handles attention type-specific constraints.
        """
        system = ConstraintSystem()
        
        # Add default constraints
        for constraint in self.info.default_constraints:
            system.add_constraint(constraint)
        
        # Add attention-specific constraints
        attention_type = spec.get('attention', 'gqa')
        
        if attention_type == 'mha':
            # Multi-head: num_kv_heads == num_heads
            system.add_constraint(EqualityConstraint(
                var1=TemplateParameter.NUM_KV_HEADS.value,
                var2=TemplateParameter.NUM_HEADS.value,
                name="mha_kv_equals_q"
            ))
        elif attention_type == 'mqa':
            # Multi-query: num_kv_heads == 1
            system.add_constraint(EqualityConstraint(
                var1=TemplateParameter.NUM_KV_HEADS.value,
                var2="1",
                name="mqa_single_kv_head"
            ))
        elif attention_type == 'gqa':
            # Grouped-query: already handled by divisibility constraint
            pass
        
        # Add user-specified constraints if any
        if spec.get('explicit_dims'):
            for param, value in spec['explicit_dims'].items():
                system.add_constraint(EqualityConstraint(
                    var1=param,
                    var2=str(value),
                    name=f"user_{param}"
                ))
        
        # Add parameter count constraint if target specified
        if 'target_params' in spec:
            system.add_constraint(EqualityConstraint(
                var1="total_params",
                var2=str(spec['target_params']),
                name="target_parameter_count"
            ))
        
        # Add activation-specific constraints
        if spec.get('activation') == 'swiglu':
            # SwiGLU intermediate size formula
            system.add_constraint(EqualityConstraint(
                var1=TemplateParameter.INTERMEDIATE_SIZE.value,
                var2=f"2 * round({TemplateParameter.HIDDEN_SIZE.value} * 8/3 / 32) * 32",
                name="swiglu_exact_intermediate"
            ))
        
        return system
    
    def calculate_parameters(self, dims: Dict[str, int]) -> int:
        """
        Calculate parameters matching build_ir():
        counts biases, all RMSNorm scales, and output projection even if tied.
        """
        n_layers = dims[TemplateParameter.NUM_LAYERS.value]
        hidden_size = dims[TemplateParameter.HIDDEN_SIZE.value]
        intermediate_size = dims[TemplateParameter.INTERMEDIATE_SIZE.value]
        vocab_size = dims[TemplateParameter.VOCAB_SIZE.value]
        num_heads = dims[TemplateParameter.NUM_HEADS.value]
        num_kv_heads = dims.get(TemplateParameter.NUM_KV_HEADS.value, num_heads)
        head_dim = dims.get(TemplateParameter.HEAD_DIM.value, hidden_size // num_heads)
        tie_embeddings = dims.get('tie_word_embeddings', True)

        total = 0

        # Embeddings
        total += vocab_size * hidden_size

        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        v_size = num_kv_heads * head_dim

        per_layer = 0
        # Q/K/V projections (weights + biases)
        per_layer += hidden_size * q_size + q_size
        per_layer += hidden_size * k_size + k_size
        per_layer += hidden_size * v_size + v_size

        # Attention output projection (weight + bias)
        per_layer += (num_heads * head_dim) * hidden_size + hidden_size

        # RMSNorms inside layer: attn_norm + mlp_norm
        per_layer += 2 * hidden_size

        # MLP projections (gate, up, down) with biases
        per_layer += hidden_size * intermediate_size + intermediate_size  # gate
        per_layer += hidden_size * intermediate_size + intermediate_size  # up
        per_layer += intermediate_size * hidden_size + hidden_size        # down

        total += per_layer * n_layers

        # Global RMSNorms: input_norm + output_norm (present when norm == rmsnorm)
        total += 2 * hidden_size

        # Output projection module; if tied, weight is shared with embeddings, so don't double-count
        if not tie_embeddings:
            total += hidden_size * vocab_size

        return total
    
    def build_ir(self, 
                 dims: Dict[str, int],
                 builder: GraphBuilder,
                 spec: Dict[str, Any]) -> IRGraph:
        """
        Build decoder-only transformer IR graph.
        """
        # Extract dimensions
        n_layers = dims[TemplateParameter.NUM_LAYERS.value]
        hidden_size = dims[TemplateParameter.HIDDEN_SIZE.value]
        intermediate_size = dims[TemplateParameter.INTERMEDIATE_SIZE.value]
        vocab_size = dims[TemplateParameter.VOCAB_SIZE.value]
        context_length = dims[TemplateParameter.CONTEXT_LENGTH.value]
        num_heads = dims[TemplateParameter.NUM_HEADS.value]
        num_kv_heads = dims.get(TemplateParameter.NUM_KV_HEADS.value, num_heads)
        head_dim = dims.get(TemplateParameter.HEAD_DIM.value, hidden_size // num_heads)
        
        # Start building graph
        graph = IRGraph(name=f"decoder_only_{self.version}")
        builder.graph = graph  # ensure builder writes into this graph
        
        # Input token IDs
        tokens = builder.create_input(
            name="input_ids",
            shape=[-1, context_length],  # batch size dynamic
            dtype="int32"
        )
        
        # Token embeddings
        embeddings = builder.create_embedding(
            name="token_embeddings",
            input=tokens,
            vocab_size=vocab_size,
            embedding_dim=hidden_size
        )
        
        # Positional embeddings
        if spec.get('positional_encoding') == 'rope':
            pos_emb = builder.create_rope(
                name="rope_positional",
                input=embeddings,
                dim=head_dim,
                theta=spec.get('rope_theta', 10000.0),
                scaling_factor=spec.get('rope_scaling_factor')
            )
            current = pos_emb
        elif spec.get('positional_encoding') == 'alibi':
            pos_emb = builder.create_alibi(
                name="alibi_bias",
                num_heads=num_heads,
                max_bias=spec.get('alibi_max_bias', 8.0)
            )
            current = embeddings
        else:
            current = embeddings
        
        # Initial normalization
        if spec.get('norm') == 'rmsnorm':
            current = builder.create_rmsnorm(
                name="input_norm",
                input=current,
                dim=hidden_size
            )
        
        # Build decoder layers
        for layer_idx in range(n_layers):
            layer_prefix = f"layer_{layer_idx}"
            
            # Attention residual path
            attn_norm = builder.create_rmsnorm(
                name=f"{layer_prefix}_attn_norm",
                input=current,
                dim=hidden_size
            )
            
            # Attention
            if spec.get('attention') in ['mha', 'gqa', 'mqa']:
                # QKV projections
                q_proj = builder.create_linear(
                    name=f"{layer_prefix}_q_proj",
                    input=attn_norm,
                    in_features=hidden_size,
                    out_features=num_heads * head_dim
                )
                
                k_proj = builder.create_linear(
                    name=f"{layer_prefix}_k_proj",
                    input=attn_norm,
                    in_features=hidden_size,
                    out_features=num_kv_heads * head_dim
                )
                
                v_proj = builder.create_linear(
                    name=f"{layer_prefix}_v_proj",
                    input=attn_norm,
                    in_features=hidden_size,
                    out_features=num_kv_heads * head_dim
                )
                
                # Multi-head attention
                attn_output = builder.create_multi_head_attention(
                    name=f"{layer_prefix}_attn",
                    query=q_proj,
                    key=k_proj,
                    value=v_proj,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    attention_type=spec.get('attention', 'gqa'),
                    use_alibi=(spec.get('positional_encoding') == 'alibi')
                )
            else:
                raise ValueError(f"Unsupported attention type: {spec.get('attention')}")
            
            # Attention output projection
            attn_out_proj = builder.create_linear(
                name=f"{layer_prefix}_attn_out_proj",
                input=attn_output,
                in_features=num_heads * head_dim,
                out_features=hidden_size
            )
            
            # First residual connection
            attn_residual = builder.create_add(
                name=f"{layer_prefix}_attn_residual",
                a=current,
                b=attn_out_proj
            )
            
            # MLP path
            mlp_norm = builder.create_rmsnorm(
                name=f"{layer_prefix}_mlp_norm",
                input=attn_residual,
                dim=hidden_size
            )
            
            # SwiGLU MLP
            if spec.get('activation') == 'swiglu':
                # Gate projection
                gate_proj = builder.create_linear(
                    name=f"{layer_prefix}_gate_proj",
                    input=mlp_norm,
                    in_features=hidden_size,
                    out_features=intermediate_size
                )
                
                # Up projection
                up_proj = builder.create_linear(
                    name=f"{layer_prefix}_up_proj",
                    input=mlp_norm,
                    in_features=hidden_size,
                    out_features=intermediate_size
                )
                
                # SwiGLU activation
                gate_act = builder.create_swiglu(
                    name=f"{layer_prefix}_swiglu",
                    gate=gate_proj,
                    up=up_proj
                )
            else:
                # Standard MLP
                fc1 = builder.create_linear(
                    name=f"{layer_prefix}_fc1",
                    input=mlp_norm,
                    in_features=hidden_size,
                    out_features=intermediate_size
                )
                
                gate_act = builder.create_activation(
                    name=f"{layer_prefix}_activation",
                    input=fc1,
                    activation_type=spec.get('activation', 'silu')
                )
            
            # Down projection
            down_proj = builder.create_linear(
                name=f"{layer_prefix}_down_proj",
                input=gate_act,
                in_features=intermediate_size,
                out_features=hidden_size
            )
            
            # Second residual connection
            current = builder.create_add(
                name=f"{layer_prefix}_mlp_residual",
                a=attn_residual,
                b=down_proj
            )
        
        # Final normalization
        if spec.get('norm') == 'rmsnorm':
            current = builder.create_rmsnorm(
                name="output_norm",
                input=current,
                dim=hidden_size
            )
        
        # Output projection (tied or separate)
        if spec.get('tie_word_embeddings', True):
            # Use embedding weights for output
            output = builder.create_linear(
                name="output_projection",
                input=current,
                in_features=hidden_size,
                out_features=vocab_size,
                use_bias=False,
                tie_weight="token_embeddings.weight"  # Reference embedding weights
            )
        else:
            # Separate output weights
            output = builder.create_linear(
                name="output_projection",
                input=current,
                in_features=hidden_size,
                out_features=vocab_size,
                use_bias=False
            )
        
        # Set as output
        builder.set_output(output, name="logits")
        
        return graph
    
    def validate_spec(self, spec: Dict[str, Any]) -> List[str]:
        """Validate specification against template requirements"""
        errors = []
        
        # Check required parameters
        if 'vocab_size' not in spec:
            errors.append("vocab_size is required")
        if 'context_length' not in spec:
            errors.append("context_length is required")
        
        # Check attention-specific requirements
        attention = spec.get('attention', 'gqa')
        if attention == 'sliding_window':
            if 'sliding_window_size' not in spec:
                errors.append("sliding_window_size required for sliding_window attention")
        
        # Check positional encoding compatibility
        pos_enc = spec.get('positional_encoding', 'rope')
        if pos_enc == 'rope':
            if attention in ['alibi', 'relative']:
                errors.append(f"rope positional encoding incompatible with {attention} attention")
        
        # Check activation function
        activation = spec.get('activation', 'swiglu')
        if activation not in ['gelu', 'relu', 'silu', 'swiglu']:
            errors.append(f"Unsupported activation: {activation}")
        
        return errors
