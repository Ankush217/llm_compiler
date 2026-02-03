"""
Validation Utilities
====================

Validation functions for specifications and generated models.
"""

from typing import Dict, Any, List, Tuple
import json
from pathlib import Path

from ..spec import LLM, NormType, AttentionType, ActivationType
from ..tokenizers.base import TokenizerTrainingStats
from ..datasets.spec import DatasetSpec

def validate_spec(
    spec: LLM,
    tokenizer_stats: TokenizerTrainingStats | None = None,
    dataset: DatasetSpec | None = None,
) -> Tuple[bool, List[str]]:
    """
    Validate LLM specification.
    
    Returns:
        (is_valid, errors)
    """
    errors = []
    
    # Check template exists
    from ..templates.registry import registry
    try:
        registry.get(spec.template)
    except ValueError:
        errors.append(f"Unknown template: {spec.template}")
    
    # Check parameter specification
    if spec.target_params is None and spec.explicit_dims is None:
        errors.append("Must specify either target_params or explicit_dims")
    
    if spec.target_params is not None and spec.target_params <= 0:
        errors.append(f"target_params must be positive: {spec.target_params}")
    if spec.target_tolerance is not None and spec.target_tolerance <= 0:
        errors.append(f"target_tolerance must be positive: {spec.target_tolerance}")
    
    # Check vocab size
    if spec.vocab_size <= 0:
        errors.append(f"vocab_size must be positive: {spec.vocab_size}")
    
    # Check context length
    if spec.context_length <= 0:
        errors.append(f"context_length must be positive: {spec.context_length}")
    
    # Check attention type
    if spec.attention not in AttentionType:
        errors.append(f"Invalid attention type: {spec.attention}")
    
    # Check norm type
    if spec.norm not in NormType:
        errors.append(f"Invalid norm type: {spec.norm}")
    
    # Check activation
    if spec.activation not in ActivationType:
        errors.append(f"Invalid activation type: {spec.activation}")
    
    # Check positional encoding
    from ..spec import PositionalEncodingType
    if spec.positional_encoding not in PositionalEncodingType:
        errors.append(f"Invalid positional encoding: {spec.positional_encoding}")
    
    # Check tokenizer
    from ..spec import TokenizerType
    if spec.tokenizer not in TokenizerType:
        errors.append(f"Invalid tokenizer: {spec.tokenizer}")
    else:
        # Prefer dataset-provided stats if present
        effective_stats = tokenizer_stats
        if dataset and dataset.tokenizer_stats:
            effective_stats = dataset.tokenizer_stats

        if effective_stats is not None:
            if effective_stats.unique_tokens < spec.vocab_size:
                errors.append(
                    f"vocab_size ({spec.vocab_size}) exceeds unique tokens in tokenizer stats "
                    f"({effective_stats.unique_tokens})"
                )
    
    # Check weight format
    from ..spec import WeightFormat
    if spec.weight_format not in WeightFormat:
        errors.append(f"Invalid weight format: {spec.weight_format}")
    
    # Check backend
    from ..spec import BackendTarget
    if spec.backend not in BackendTarget:
        errors.append(f"Invalid backend: {spec.backend}")
    
    # Check precision
    if spec.precision not in ['float32', 'float16', 'bfloat16']:
        errors.append(f"Invalid precision: {spec.precision}")
    
    # Check RoPE scaling
    if (spec.positional_encoding == PositionalEncodingType.ROPE and 
        spec.rope_scaling_factor is not None and
        spec.rope_scaling_factor <= 0):
        errors.append(f"rope_scaling_factor must be positive: {spec.rope_scaling_factor}")
    
    # Check sliding window
    if (spec.attention == AttentionType.SLIDING_WINDOW and
        spec.sliding_window_size is None):
        errors.append("sliding_window_size required for sliding_window attention")
    
    if (spec.sliding_window_size is not None and
        spec.sliding_window_size > spec.context_length):
        errors.append(f"sliding_window_size ({spec.sliding_window_size}) "
                     f"exceeds context_length ({spec.context_length})")
    
    return len(errors) == 0, errors

def validate_generated_model(model_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate generated model files.
    
    Args:
        model_dir: Directory containing generated model
        
    Returns:
        (is_valid, errors)
    """
    errors = []
    
    # Check required files
    required_files = [
        "spec.json",
        "solution.json",
        "compilation_report.json",
        "model/__init__.py",
        "model/config.py",
    ]
    
    for file in required_files:
        if not (model_dir / file).exists():
            errors.append(f"Missing required file: {file}")
    
    # Check spec and solution consistency
    try:
        spec_path = model_dir / "spec.json"
        solution_path = model_dir / "solution.json"
        ir_path = model_dir / "ir_graph.json"
        
        spec = json.loads(spec_path.read_text())
        solution = json.loads(solution_path.read_text())
        
        # Check parameter count matches
        if 'target_params' in spec:
            target = spec['target_params']
            actual = solution.get('parameters', 0)
            
            if abs(actual - target) / target > 0.01:  # 1% tolerance
                errors.append(f"Parameter mismatch: target={target:,}, actual={actual:,}")

        # Cross-validate against IR parameter count if available
        if ir_path.exists():
            from ..ir.graph import IRGraph
            from ..utils.parameters import count_parameters_from_ir
            ir_graph = IRGraph.from_json(ir_path.read_text())
            ir_params = count_parameters_from_ir(ir_graph)
            if 'parameters' in solution and solution['parameters'] != ir_params:
                errors.append(
                    f"IR parameter count {ir_params:,} disagrees with solution {solution['parameters']:,}"
                )
    
    except Exception as e:
        errors.append(f"Error reading spec/solution: {e}")
    
    # Check model code compiles
    try:
        model_files = list((model_dir / "model").glob("*.py"))
        if not model_files:
            errors.append("No Python model files found")
    except:
        errors.append("Model directory not found")
    
    return len(errors) == 0, errors

def validate_ir_graph(graph_file: Path) -> Tuple[bool, List[str]]:
    """
    Validate IR graph file.
    
    Args:
        graph_file: IR graph JSON file
        
    Returns:
        (is_valid, errors)
    """
    errors = []
    
    try:
        from ..ir.graph import IRGraph
        graph = IRGraph.from_json(graph_file.read_text())
        
        # Run graph validation
        graph_errors = graph.validate()
        errors.extend(graph_errors)
        
        # Check for cycles (simplified)
        # In full implementation would do topological sort
        
    except Exception as e:
        errors.append(f"Error loading IR graph: {e}")
    
    return len(errors) == 0, errors
