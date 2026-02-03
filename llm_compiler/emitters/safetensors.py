"""
Safetensors Emitter
===================

Emits model weights in safetensors format.
Generates both weights and metadata.
"""

from typing import Dict, List, Any, Optional
import json
import struct
import numpy as np
from pathlib import Path

class SafetensorsEmitter:
    """Emits weights in safetensors format"""
    
    def __init__(self):
        self.header_size = 8  # bytes for header length
        
    def emit(self,
             weights: Dict[str, np.ndarray],
             metadata: Dict[str, Any],
             output_path: Path) -> Dict[str, Any]:
        """
        Emit weights in safetensors format.
        
        Args:
            weights: Dictionary of tensor name -> numpy array
            metadata: Model metadata
            output_path: Output file path
            
        Returns:
            Dictionary with emission info
        """
        # Prepare tensors
        tensor_data = {}
        offsets = {}
        
        current_offset = 0
        
        # Calculate offsets
        for name, tensor in weights.items():
            # Convert to contiguous array
            tensor = np.ascontiguousarray(tensor)
            
            # Store data
            tensor_data[name] = tensor
            
            # Calculate offset and size
            size = tensor.nbytes
            dtype = self._numpy_to_safetensors_dtype(tensor.dtype)
            shape = list(tensor.shape)
            
            offsets[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [current_offset, current_offset + size]
            }
            
            current_offset += size
        
        # Create header
        header = {
            "__metadata__": metadata
        }
        header.update(offsets)
        
        # Serialize header
        header_json = json.dumps(header).encode('utf-8')
        header_length = len(header_json)
        
        # Calculate padding
        total_header_size = self.header_size + header_length
        pad = 8 - (total_header_size % 8)
        if pad == 8:
            pad = 0
        
        # Write file
        with open(output_path, 'wb') as f:
            # Write header length
            f.write(struct.pack('<Q', header_length))
            
            # Write header
            f.write(header_json)
            
            # Write padding
            if pad > 0:
                f.write(b'\x00' * pad)
            
            # Write tensor data
            for name in weights.keys():
                tensor = tensor_data[name]
                f.write(tensor.tobytes())
        
        # Return info
        return {
            "file_path": str(output_path),
            "file_size": output_path.stat().st_size,
            "num_tensors": len(weights),
            "total_bytes": current_offset,
            "header_size": total_header_size + pad,
            "metadata": metadata
        }
    
    def _numpy_to_safetensors_dtype(self, dtype: np.dtype) -> str:
        """Convert numpy dtype to safetensors dtype string"""
        mapping = {
            np.float32: "F32",
            np.float16: "F16",
            np.bfloat16: "BF16",
            np.int32: "I32",
            np.int64: "I64",
            np.uint8: "U8",
            np.bool_: "BOOL",
        }
        
        for np_type, st_type in mapping.items():
            if np.issubdtype(dtype, np_type):
                return st_type
        
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    def create_metadata(self,
                       model_name: str,
                       config: Dict[str, Any],
                       parameter_count: int) -> Dict[str, Any]:
        """Create safetensors metadata"""
        return {
            "model_name": model_name,
            "format": "pytorch",
            "architecture": config.get('template', 'unknown'),
            "vocab_size": config.get('vocab_size', 0),
            "hidden_size": config.get('hidden_size', 0),
            "num_layers": config.get('num_layers', 0),
            "num_attention_heads": config.get('num_attention_heads', 0),
            "num_key_value_heads": config.get('num_key_value_heads', 0),
            "head_dim": config.get('head_dim', 0),
            "intermediate_size": config.get('intermediate_size', 0),
            "max_position_embeddings": config.get('context_length', 0),
            "rope_theta": config.get('rope_theta', 10000.0),
            "attention_type": config.get('attention_type', 'gqa'),
            "norm_type": config.get('norm', 'rmsnorm'),
            "activation": config.get('activation', 'swiglu'),
            "parameter_count": parameter_count,
            "generator": "llm_compiler",
            "version": "1.0.0"
        }
    
    def generate_weight_names(self, 
                             model_name: str,
                             num_layers: int) -> List[str]:
        """Generate weight names for model"""
        names = []
        
        # Embeddings
        names.append(f"{model_name}.embed_tokens.weight")
        
        # Layers
        for i in range(num_layers):
            prefix = f"{model_name}.layers.{i}"
            
            # Attention
            names.append(f"{prefix}.self_attn.q_proj.weight")
            names.append(f"{prefix}.self_attn.k_proj.weight")
            names.append(f"{prefix}.self_attn.v_proj.weight")
            names.append(f"{prefix}.self_attn.o_proj.weight")
            
            # Norms
            names.append(f"{prefix}.input_layernorm.weight")
            names.append(f"{prefix}.post_attention_layernorm.weight")
            
            # MLP
            names.append(f"{prefix}.mlp.w1.weight")
            names.append(f"{prefix}.mlp.w2.weight")
            names.append(f"{prefix}.mlp.w3.weight")
        
        # Final norm
        names.append(f"{model_name}.norm.weight")
        
        # LM head (if not tied)
        names.append(f"{model_name}.lm_head.weight")
        
        return names