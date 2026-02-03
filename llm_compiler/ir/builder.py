"""
IR Graph Builder
================

Helper for building IR graphs from templates.
Provides high-level operations that map to IR nodes.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from .graph import IRGraph, Tensor, Operation, NodeType, TensorDType

class GraphBuilder:
    """Builder for IR graphs"""
    
    def __init__(self, graph: IRGraph = None):
        self.graph = graph or IRGraph("unnamed")
        self._tensor_counter = 0
        self._op_counter = 0
    
    def _new_name(self, prefix: str) -> str:
        """Generate unique name"""
        self._tensor_counter += 1
        return f"{prefix}_{self._tensor_counter}"
    
    def _new_op_name(self, prefix: str) -> str:
        """Generate unique operation name"""
        self._op_counter += 1
        return f"{prefix}_{self._op_counter}"
    
    def create_input(self, 
                     name: str,
                     shape: List[Any],
                     dtype: Union[str, TensorDType] = "float32") -> str:
        """Create input tensor"""
        if isinstance(dtype, str):
            dtype = TensorDType(dtype)
        
        tensor = Tensor(
            name=name,
            shape=shape,
            dtype=dtype
        )
        
        self.graph.add_tensor(tensor)
        self.graph.add_input(name)
        return name
    
    def create_constant(self,
                        name: str,
                        value: Any,
                        shape: List[int],
                        dtype: Union[str, TensorDType] = "float32") -> str:
        """Create constant tensor"""
        if isinstance(dtype, str):
            dtype = TensorDType(dtype)
        
        tensor = Tensor(
            name=name,
            shape=shape,
            dtype=dtype
        )
        
        op = Operation(
            name=self._new_op_name("constant"),
            node_type=NodeType.CONSTANT,
            inputs=[],
            outputs=[name],
            attributes={
                "value": value,
                "shape": shape,
                "dtype": dtype.value
            }
        )
        
        self.graph.add_tensor(tensor)
        self.graph.add_operation(op)
        return name
    
    def create_embedding(self,
                        name: str,
                        input: str,
                        vocab_size: int,
                        embedding_dim: int) -> str:
        """Create embedding operation"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.EMBEDDING,
            inputs=[input],
            outputs=[output],
            attributes={
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim
            }
        )
        
        self.graph.add_operation(op)
        # Record tensor shape/dtype for downstream semantic validation
        self.graph.tensors[output].shape = [ -1, -1, embedding_dim ]
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_linear(self,
                     name: str,
                     input: str,
                     in_features: int,
                     out_features: int,
                     use_bias: bool = True,
                     tie_weight: Optional[str] = None) -> str:
        """Create linear (fully connected) layer"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.LINEAR,
            inputs=[input],
            outputs=[output],
            attributes={
                "in_features": in_features,
                "out_features": out_features,
                "use_bias": use_bias,
                "tie_weight": tie_weight
            }
        )
        
        self.graph.add_operation(op)
        # Assume last dimension equals out_features; batch/sequence preserved
        input_shape = self.graph.tensors.get(input, Tensor("", [], TensorDType.FLOAT32)).shape
        prefix = input_shape[:-1] if input_shape else [-1]
        self.graph.tensors[output].shape = prefix + [out_features]
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_rmsnorm(self,
                      name: str,
                      input: str,
                      dim: int) -> str:
        """Create RMSNorm operation"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.RMSNORM,
            inputs=[input],
            outputs=[output],
            attributes={
                "normalized_shape": dim,
                "eps": 1e-6
            }
        )
        
        self.graph.add_operation(op)
        # RMSNorm preserves shape
        self.graph.tensors[output].shape = list(self.graph.tensors.get(input, Tensor("", [], TensorDType.FLOAT32)).shape)
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_layernorm(self,
                        name: str,
                        input: str,
                        dim: int,
                        eps: float = 1e-5) -> str:
        """Create LayerNorm operation"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.LAYERNORM,
            inputs=[input],
            outputs=[output],
            attributes={
                "normalized_shape": dim,
                "eps": eps
            }
        )
        
        self.graph.add_operation(op)
        self.graph.tensors[output].shape = list(self.graph.tensors.get(input, Tensor("", [], TensorDType.FLOAT32)).shape)
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_rope(self,
                   name: str,
                   input: str,
                   dim: int,
                   theta: float = 10000.0,
                   scaling_factor: Optional[float] = None) -> str:
        """Create RoPE (Rotary Positional Embedding)"""
        output = self._new_name(f"{name}_out")
        
        attrs = {
            "dim": dim,
            "theta": theta
        }
        if scaling_factor is not None:
            attrs["scaling_factor"] = scaling_factor
        
        op = Operation(
            name=name,
            node_type=NodeType.ROPE,
            inputs=[input],
            outputs=[output],
            attributes=attrs
        )
        
        self.graph.add_operation(op)
        # RoPE preserves shape
        self.graph.tensors[output].shape = list(self.graph.tensors.get(input, Tensor("", [], TensorDType.FLOAT32)).shape)
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_alibi(self,
                    name: str,
                    num_heads: int,
                    max_bias: float = 8.0) -> str:
        """Create ALiBi (Attention with Linear Biases)"""
        output = self._new_name(f"{name}_bias")
        
        op = Operation(
            name=name,
            node_type=NodeType.ALIBI,
            inputs=[],
            outputs=[output],
            attributes={
                "num_heads": num_heads,
                "max_bias": max_bias
            }
        )
        
        self.graph.add_operation(op)
        # ALiBi bias broadcast shape
        self.graph.tensors[output].shape = [1, num_heads, 1, 1]
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_multi_head_attention(self,
                                   name: str,
                                   query: str,
                                   key: str,
                                   value: str,
                                   num_heads: int,
                                   num_kv_heads: int,
                                   head_dim: int,
                                   attention_type: str = "gqa",
                                   use_alibi: bool = False) -> str:
        """Create multi-head attention operation"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.MULTI_HEAD_ATTENTION,
            inputs=[query, key, value],
            outputs=[output],
            attributes={
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "attention_type": attention_type,
                "use_alibi": use_alibi
            }
        )
        
        self.graph.add_operation(op)
        # Output shape matches query with hidden dim = num_heads * head_dim
        q_shape = self.graph.tensors.get(query, Tensor("", [], TensorDType.FLOAT32)).shape
        prefix = q_shape[:-1] if q_shape else [-1, -1]
        self.graph.tensors[output].shape = prefix + [num_heads * head_dim]
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_swiglu(self,
                     name: str,
                     gate: str,
                     up: str) -> str:
        """Create SwiGLU activation"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.SWIGLU,
            inputs=[gate, up],
            outputs=[output],
            attributes={}
        )
        
        self.graph.add_operation(op)
        # Output shape follows gate/up input shape
        self.graph.tensors[output].shape = list(self.graph.tensors.get(gate, Tensor("", [], TensorDType.FLOAT32)).shape)
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_activation(self,
                         name: str,
                         input: str,
                         activation_type: str) -> str:
        """Create activation function"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.ACTIVATION,
            inputs=[input],
            outputs=[output],
            attributes={
                "activation": activation_type
            }
        )
        
        self.graph.add_operation(op)
        self.graph.tensors[output].shape = list(self.graph.tensors.get(input, Tensor("", [], TensorDType.FLOAT32)).shape)
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_add(self,
                  name: str,
                  a: str,
                  b: str) -> str:
        """Create element-wise addition"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.ADD,
            inputs=[a, b],
            outputs=[output],
            attributes={}
        )
        
        self.graph.add_operation(op)
        # Addition preserves shape (assume broadcast-compatible)
        self.graph.tensors[output].shape = list(self.graph.tensors.get(a, Tensor("", [], TensorDType.FLOAT32)).shape)
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_mul(self,
                  name: str,
                  a: str,
                  b: str) -> str:
        """Create element-wise multiplication"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.MUL,
            inputs=[a, b],
            outputs=[output],
            attributes={}
        )
        
        self.graph.add_operation(op)
        self.graph.tensors[output].shape = list(self.graph.tensors.get(a, Tensor("", [], TensorDType.FLOAT32)).shape)
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_matmul(self,
                     name: str,
                     a: str,
                     b: str,
                     transpose_a: bool = False,
                     transpose_b: bool = False) -> str:
        """Create matrix multiplication"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.MATMUL,
            inputs=[a, b],
            outputs=[output],
            attributes={
                "transpose_a": transpose_a,
                "transpose_b": transpose_b
            }
        )
        
        self.graph.add_operation(op)
        self.graph.tensors[output].shape = []
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_softmax(self,
                      name: str,
                      input: str,
                      dim: int = -1) -> str:
        """Create softmax operation"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.SOFTMAX,
            inputs=[input],
            outputs=[output],
            attributes={
                "dim": dim
            }
        )
        
        self.graph.add_operation(op)
        self.graph.tensors[output].shape = list(self.graph.tensors.get(input, Tensor("", [], TensorDType.FLOAT32)).shape)
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def create_dropout(self,
                      name: str,
                      input: str,
                      rate: float = 0.1) -> str:
        """Create dropout operation"""
        output = self._new_name(f"{name}_out")
        
        op = Operation(
            name=name,
            node_type=NodeType.DROPOUT,
            inputs=[input],
            outputs=[output],
            attributes={
                "rate": rate
            }
        )
        
        self.graph.add_operation(op)
        self.graph.tensors[output].shape = list(self.graph.tensors.get(input, Tensor("", [], TensorDType.FLOAT32)).shape)
        self.graph.tensors[output].dtype = TensorDType.FLOAT32
        return output
    
    def set_output(self, tensor: str, name: str = None):
        """Mark tensor as graph output"""
        self.graph.add_output(tensor)
    
    def get_graph(self) -> IRGraph:
        """Get built graph"""
        return self.graph
