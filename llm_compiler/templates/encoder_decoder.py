"""
Encoder-Decoder Transformer Template
====================================

Implements T5/BART-style encoder-decoder architectures.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from .base import ArchitectureTemplate, TemplateInfo, TemplateParameter
from ..ir.graph import IRGraph
from ..ir.builder import GraphBuilder
from ..solver.constraints import (
    ConstraintSystem, EqualityConstraint, 
    DivisibilityConstraint, RangeConstraint
)

class EncoderDecoderTemplate(ArchitectureTemplate):
    """Encoder-decoder transformer (T5/BART style)"""
    
    def __init__(self, version: str = "v1"):
        self.version = version
        
    @property
    def info(self) -> TemplateInfo:
        return TemplateInfo(
            name=f"encoder_decoder_{self.version}",
            description="Encoder-decoder transformer (T5/BART style)",
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
                "num_encoder_layers",
                "num_decoder_layers",
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
                "num_encoder_layers",
                "num_decoder_layers",
            },
            default_constraints=[
                # Shared constraints
                DivisibilityConstraint(
                    var1=TemplateParameter.HIDDEN_SIZE.value,
                    var2=TemplateParameter.NUM_HEADS.value,
                    name="hidden_size_divisible_by_num_heads"
                ),
                # Encoder-decoder specific
                EqualityConstraint(
                    var1="num_layers",
                    var2="num_encoder_layers + num_decoder_layers",
                    name="total_layers_sum"
                ),
            ]
        )
    
    def create_constraint_system(self, spec: Dict[str, Any]) -> ConstraintSystem:
        """Create constraint system for encoder-decoder architecture"""
        system = ConstraintSystem()
        
        # Add default constraints
        for constraint in self.info.default_constraints:
            system.add_constraint(constraint)
        
        # Encoder-decoder typically uses same hidden size throughout
        system.add_constraint(EqualityConstraint(
            var1="encoder_hidden_size",
            var2=TemplateParameter.HIDDEN_SIZE.value,
            name="encoder_hidden_size_equals_main"
        ))
        
        system.add_constraint(EqualityConstraint(
            var1="decoder_hidden_size",
            var2=TemplateParameter.HIDDEN_SIZE.value,
            name="decoder_hidden_size_equals_main"
        ))
        
        # Add user constraints
        if 'explicit_dims' in spec:
            for param, value in spec['explicit_dims'].items():
                system.add_constraint(EqualityConstraint(
                    var1=param,
                    var2=str(value),
                    name=f"user_{param}"
                ))
        
        if 'target_params' in spec:
            system.add_constraint(EqualityConstraint(
                var1="total_params",
                var2=str(spec['target_params']),
                name="target_parameter_count"
            ))
        
        return system
    
    def calculate_parameters(self, dims: Dict[str, int]) -> int:
        """Calculate parameters for encoder-decoder transformer"""
        # Similar to decoder-only but with encoder and cross-attention
        hidden_size = dims[TemplateParameter.HIDDEN_SIZE.value]
        intermediate_size = dims[TemplateParameter.INTERMEDIATE_SIZE.value]
        vocab_size = dims[TemplateParameter.VOCAB_SIZE.value]
        num_heads = dims[TemplateParameter.NUM_HEADS.value]
        head_dim = dims.get(TemplateParameter.HEAD_DIM.value, hidden_size // num_heads)
        
        # Get layer counts
        num_encoder_layers = dims.get('num_encoder_layers', dims.get('num_layers', 12) // 2)
        num_decoder_layers = dims.get('num_decoder_layers', dims.get('num_layers', 12) // 2)
        
        # Encoder parameters (similar to decoder-only but without cross-attention)
        encoder_per_layer = self._calculate_encoder_layer_params(
            hidden_size, intermediate_size, num_heads, head_dim
        )
        
        # Decoder parameters (with cross-attention)
        decoder_per_layer = self._calculate_decoder_layer_params(
            hidden_size, intermediate_size, num_heads, head_dim
        )
        
        # Total
        total = (encoder_per_layer * num_encoder_layers +
                 decoder_per_layer * num_decoder_layers)
        
        # Embeddings (shared typically)
        total += vocab_size * hidden_size
        
        return total
    
    def _calculate_encoder_layer_params(self, hidden_size, intermediate_size, num_heads, head_dim):
        """Calculate parameters for one encoder layer"""
        params = 0
        
        # Self-attention
        params += 4 * hidden_size * hidden_size  # QKV + output
        params += hidden_size  # RMSNorm
        
        # MLP
        params += 2 * hidden_size * intermediate_size  # gate/up
        params += intermediate_size * hidden_size  # down
        params += hidden_size  # RMSNorm
        
        return params
    
    def _calculate_decoder_layer_params(self, hidden_size, intermediate_size, num_heads, head_dim):
        """Calculate parameters for one decoder layer"""
        params = self._calculate_encoder_layer_params(hidden_size, intermediate_size, num_heads, head_dim)
        
        # Add cross-attention
        params += 4 * hidden_size * hidden_size  # Cross-attention QKV + output
        params += hidden_size  # Additional RMSNorm for cross-attention
        
        return params
    
    def build_ir(self, dims: Dict[str, int], builder: GraphBuilder, spec: Dict[str, Any]) -> IRGraph:
        """Build encoder-decoder IR graph"""
        # Implementation similar to decoder-only but with encoder/decoder separation
        # This would be quite long - showing structure
        graph = IRGraph(name=f"encoder_decoder_{self.version}")
        
        # Build encoder
        # Build decoder with cross-attention
        # Connect them
        
        return graph
    
    def validate_spec(self, spec: Dict[str, Any]) -> List[str]:
        """Validate encoder-decoder spec"""
        errors = []
        
        if 'vocab_size' not in spec:
            errors.append("vocab_size is required")
        if 'context_length' not in spec:
            errors.append("context_length is required")
        
        return errors