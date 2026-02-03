"""
Intermediate Representation (IR)
================================

Framework-agnostic graph representation of LLM architectures.
Explicit tensors, shapes, and connections.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

class TensorDType(Enum):
    """Tensor data types"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"

class NodeType(Enum):
    """IR node types"""
    INPUT = "input"
    OUTPUT = "output"
    CONSTANT = "constant"
    
    # Operations
    LINEAR = "linear"
    EMBEDDING = "embedding"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    ATTENTION = "attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    ROPE = "rope"
    ALIBI = "alibi"
    SWIGLU = "swiglu"
    ACTIVATION = "activation"
    ADD = "add"
    MUL = "mul"
    MATMUL = "matmul"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    CONCAT = "concat"
    SPLIT = "split"
    SLICE = "slice"
    GATHER = "gather"
    SOFTMAX = "softmax"
    DROPOUT = "dropout"
    
    # Control flow
    LOOP = "loop"
    CONDITIONAL = "conditional"

@dataclass
class Tensor:
    """IR Tensor representation"""
    name: str
    shape: List[Any]  # Can contain -1 for dynamic dimensions
    dtype: TensorDType
    node: Optional[str] = None  # Producing node
    consumers: List[str] = field(default_factory=list)  # Consuming nodes
    
    def __str__(self) -> str:
        shape_str = "[" + ", ".join(str(s) for s in self.shape) + "]"
        return f"{self.name}: {shape_str} {self.dtype.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype.value,
            "node": self.node,
            "consumers": self.consumers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tensor':
        return cls(
            name=data["name"],
            shape=data["shape"],
            dtype=TensorDType(data["dtype"]),
            node=data.get("node"),
            consumers=data.get("consumers", [])
        )

@dataclass
class Operation:
    """IR Operation node"""
    name: str
    node_type: NodeType
    inputs: List[str]  # Input tensor names
    outputs: List[str]  # Output tensor names
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        inputs_str = ", ".join(self.inputs)
        outputs_str = ", ".join(self.outputs)
        attrs_str = ""
        if self.attributes:
            attrs_str = " " + " ".join(f"{k}={v}" for k, v in self.attributes.items())
        return f"{self.name}: {self.node_type.value}({inputs_str}) -> {outputs_str}{attrs_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.node_type.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        return cls(
            name=data["name"],
            node_type=NodeType(data["type"]),
            inputs=data["inputs"],
            outputs=data["outputs"],
            attributes=data.get("attributes", {})
        )

class IRGraph:
    """Complete IR Graph"""
    
    def __init__(self, name: str):
        self.name = name
        self.tensors: Dict[str, Tensor] = {}
        self.operations: Dict[str, Operation] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []
    
    def add_tensor(self, tensor: Tensor):
        """Add tensor to graph"""
        if tensor.name in self.tensors:
            raise ValueError(f"Tensor {tensor.name} already exists")
        self.tensors[tensor.name] = tensor
    
    def add_operation(self, operation: Operation):
        """Add operation to graph"""
        if operation.name in self.operations:
            raise ValueError(f"Operation {operation.name} already exists")
        
        # Update tensor references
        for output in operation.outputs:
            if output in self.tensors:
                self.tensors[output].node = operation.name
            else:
                # Create output tensor
                self.tensors[output] = Tensor(
                    name=output,
                    shape=[],  # Unknown shape
                    dtype=TensorDType.FLOAT32,  # Default
                    node=operation.name
                )
        
        # Update input tensor consumers
        for input_name in operation.inputs:
            if input_name in self.tensors:
                self.tensors[input_name].consumers.append(operation.name)
            else:
                # Create input tensor if it doesn't exist
                self.tensors[input_name] = Tensor(
                    name=input_name,
                    shape=[],  # Unknown shape
                    dtype=TensorDType.FLOAT32,  # Default
                    consumers=[operation.name]
                )
        
        self.operations[operation.name] = operation
    
    def add_input(self, tensor_name: str):
        """Mark tensor as graph input"""
        if tensor_name not in self.inputs:
            self.inputs.append(tensor_name)
    
    def add_output(self, tensor_name: str):
        """Mark tensor as graph output"""
        if tensor_name not in self.outputs:
            self.outputs.append(tensor_name)
    
    def validate(self) -> List[str]:
        """Validate graph consistency"""
        errors = []
        
        # Check all tensors have producers or are inputs
        for tensor_name, tensor in self.tensors.items():
            if tensor.node is None and tensor_name not in self.inputs:
                errors.append(f"Tensor {tensor_name} has no producer and is not an input")
            
            # Check consumers exist
            for consumer in tensor.consumers:
                if consumer not in self.operations:
                    errors.append(f"Tensor {tensor_name} references non-existent consumer {consumer}")
        
        # Check operations reference valid tensors
        for op_name, operation in self.operations.items():
            for input_name in operation.inputs:
                if input_name not in self.tensors:
                    errors.append(f"Operation {op_name} references non-existent input {input_name}")
            
            for output_name in operation.outputs:
                if output_name not in self.tensors:
                    errors.append(f"Operation {op_name} references non-existent output {output_name}")
        
        # Check outputs exist
        for output_name in self.outputs:
            if output_name not in self.tensors:
                errors.append(f"Output {output_name} does not exist")

        # Semantic validation (shapes/dtypes)
        errors.extend(self.validate_semantics())

        return errors

    def _topological_order(self) -> List[str]:
        """Return operation names in topological order"""
        indegree = {name: 0 for name in self.operations}
        for op in self.operations.values():
            for inp in op.inputs:
                producer = self.tensors.get(inp, Tensor("", [], TensorDType.FLOAT32)).node
                if producer and producer in indegree:
                    indegree[op.name] += 1
        ready = [name for name, deg in indegree.items() if deg == 0]
        order = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            op = self.operations[current]
            for out in op.outputs:
                for consumer in self.tensors.get(out, Tensor("", [], TensorDType.FLOAT32)).consumers:
                    if consumer in indegree:
                        indegree[consumer] -= 1
                        if indegree[consumer] == 0:
                            ready.append(consumer)
        # Fallback to insertion order if cycle detected
        if len(order) != len(self.operations):
            return list(self.operations.keys())
        return order

    def validate_semantics(self) -> List[str]:
        """Validate shapes/dtypes for common operations"""
        errors: List[str] = []
        tensor_shapes = {name: tensor.shape for name, tensor in self.tensors.items()}
        tensor_dtypes = {name: tensor.dtype for name, tensor in self.tensors.items()}

        def _rank(shape):
            return len(shape) if shape is not None else 0

        def _dim_equal(a, b):
            return a == b or a == -1 or b == -1

        def _propagate_output(op_name, output_name, shape, dtype=TensorDType.FLOAT32):
            if shape is not None:
                tensor_shapes[output_name] = list(shape)
            tensor_dtypes[output_name] = dtype

        for op_name in self._topological_order():
            op = self.operations[op_name]
            try:
                if op.node_type == NodeType.LINEAR:
                    in_tensor = op.inputs[0]
                    in_shape = tensor_shapes.get(in_tensor)
                    in_features = op.attributes.get("in_features")
                    out_features = op.attributes.get("out_features")
                    if in_shape and in_features is not None and len(in_shape) > 0:
                        if in_shape[-1] != in_features:
                            errors.append(
                                f"{op_name}: input last dim {in_shape[-1]} "
                                f"does not match linear in_features {in_features}"
                            )
                    if out_features is not None:
                        prefix = in_shape[:-1] if in_shape else [-1]
                        _propagate_output(op_name, op.outputs[0], prefix + [out_features])

                elif op.node_type in (NodeType.RMSNORM, NodeType.LAYERNORM):
                    src = op.inputs[0]
                    in_shape = tensor_shapes.get(src)
                    normalized_shape = op.attributes.get("normalized_shape")
                    if in_shape and normalized_shape is not None:
                        if in_shape[-1] != normalized_shape:
                            errors.append(
                                f"{op_name}: normalized_shape {normalized_shape} "
                                f"does not match input hidden dim {in_shape[-1]}"
                            )
                    _propagate_output(op_name, op.outputs[0], in_shape)

                elif op.node_type == NodeType.EMBEDDING:
                    input_name = op.inputs[0]
                    in_shape = tensor_shapes.get(input_name, [])
                    vocab = op.attributes.get("vocab_size")
                    if vocab is None:
                        errors.append(f"{op_name}: missing vocab_size attribute")
                    _propagate_output(op_name, op.outputs[0], in_shape + [op.attributes.get("embedding_dim", 0)])

                elif op.node_type == NodeType.MULTI_HEAD_ATTENTION:
                    q_name, k_name, v_name = op.inputs[:3]
                    q_shape = tensor_shapes.get(q_name)
                    k_shape = tensor_shapes.get(k_name)
                    v_shape = tensor_shapes.get(v_name)
                    num_heads = op.attributes.get("num_heads")
                    num_kv_heads = op.attributes.get("num_kv_heads", num_heads)
                    head_dim = op.attributes.get("head_dim")
                    if q_shape and num_heads and head_dim and q_shape[-1] != num_heads * head_dim:
                        errors.append(
                            f"{op_name}: query hidden dim {q_shape[-1]} != num_heads*head_dim ({num_heads*head_dim})"
                        )
                    if k_shape and num_kv_heads and head_dim and k_shape[-1] != num_kv_heads * head_dim:
                        errors.append(
                            f"{op_name}: key hidden dim {k_shape[-1]} != num_kv_heads*head_dim ({num_kv_heads*head_dim})"
                        )
                    if v_shape and num_kv_heads and head_dim and v_shape[-1] != num_kv_heads * head_dim:
                        errors.append(
                            f"{op_name}: value hidden dim {v_shape[-1]} != num_kv_heads*head_dim ({num_kv_heads*head_dim})"
                        )
                    # Batch/seq consistency
                    if q_shape and k_shape and _rank(q_shape) >= 2 and _rank(k_shape) >= 2:
                        if not _dim_equal(q_shape[0], k_shape[0]):
                            errors.append(f"{op_name}: batch mismatch q({q_shape[0]}) vs k({k_shape[0]})")
                    if q_shape and v_shape and _rank(q_shape) >= 2 and _rank(v_shape) >= 2:
                        if not _dim_equal(q_shape[0], v_shape[0]):
                            errors.append(f"{op_name}: batch mismatch q({q_shape[0]}) vs v({v_shape[0]})")
                    # Optional attention mask as 4th input
                    if len(op.inputs) >= 4:
                        mask_shape = tensor_shapes.get(op.inputs[3])
                        if mask_shape and _rank(mask_shape) >= 4 and q_shape:
                            if not _dim_equal(mask_shape[0], q_shape[0]):
                                errors.append(f"{op_name}: attention mask batch {mask_shape[0]} != query batch {q_shape[0]}")
                            if not _dim_equal(mask_shape[-1], k_shape[1] if k_shape and _rank(k_shape) >= 2 else -1):
                                errors.append(f"{op_name}: attention mask length {mask_shape[-1]} incompatible with key seq")
                    # Output shape follows query
                    if q_shape:
                        _propagate_output(op_name, op.outputs[0], q_shape)

                elif op.node_type in (NodeType.ADD, NodeType.MUL):
                    a_shape = tensor_shapes.get(op.inputs[0])
                    b_shape = tensor_shapes.get(op.inputs[1])
                    if a_shape and b_shape and a_shape != b_shape:
                        errors.append(f"{op_name}: operand shape mismatch {a_shape} vs {b_shape}")
                    _propagate_output(op_name, op.outputs[0], a_shape or b_shape or [])

                elif op.node_type == NodeType.SWIGLU:
                    gate_shape = tensor_shapes.get(op.inputs[0])
                    up_shape = tensor_shapes.get(op.inputs[1])
                    if gate_shape and up_shape and gate_shape != up_shape:
                        errors.append(f"{op_name}: gate/up shapes differ {gate_shape} vs {up_shape}")
                    _propagate_output(op_name, op.outputs[0], gate_shape or up_shape or [])

                elif op.node_type == NodeType.ACTIVATION:
                    src = op.inputs[0]
                    _propagate_output(op_name, op.outputs[0], tensor_shapes.get(src, []), tensor_dtypes.get(src, TensorDType.FLOAT32))

                elif op.node_type == NodeType.MATMUL:
                    a_shape = tensor_shapes.get(op.inputs[0], [])
                    b_shape = tensor_shapes.get(op.inputs[1], [])
                    if _rank(a_shape) >= 2 and _rank(b_shape) >= 2:
                        if not _dim_equal(a_shape[-1], b_shape[-2]):
                            errors.append(f"{op_name}: matmul dim mismatch {a_shape[-1]} vs {b_shape[-2]}")
                        out_shape = a_shape[:-1] + [b_shape[-1]]
                        _propagate_output(op_name, op.outputs[0], out_shape)
                    else:
                        _propagate_output(op_name, op.outputs[0], [])
                else:
                    # Default: propagate first input shape if available
                    if op.inputs:
                        _propagate_output(op_name, op.outputs[0], tensor_shapes.get(op.inputs[0], []), tensor_dtypes.get(op.inputs[0], TensorDType.FLOAT32))
            except Exception as exc:
                errors.append(f"{op_name}: semantic validation error {exc}")

        return errors
    
    def get_parameter_count(self) -> int:
        """Estimate parameter count from graph"""
        from ..utils.parameters import count_parameters_from_ir
        return count_parameters_from_ir(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary"""
        return {
            "name": self.name,
            "tensors": {name: tensor.to_dict() for name, tensor in self.tensors.items()},
            "operations": {name: op.to_dict() for name, op in self.operations.items()},
            "inputs": self.inputs,
            "outputs": self.outputs
        }
    
    def to_json(self) -> str:
        """Serialize graph to JSON"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRGraph':
        """Deserialize graph from dictionary"""
        graph = cls(data["name"])
        
        # Load tensors
        for name, tensor_data in data.get("tensors", {}).items():
            graph.tensors[name] = Tensor.from_dict(tensor_data)
        
        # Load operations
        for name, op_data in data.get("operations", {}).items():
            graph.operations[name] = Operation.from_dict(op_data)
        
        graph.inputs = data.get("inputs", [])
        graph.outputs = data.get("outputs", [])
        
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> 'IRGraph':
        """Deserialize graph from JSON"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        lines = [f"IRGraph: {self.name}"]
        lines.append(f"  Tensors: {len(self.tensors)}")
        lines.append(f"  Operations: {len(self.operations)}")
        lines.append(f"  Inputs: {self.inputs}")
        lines.append(f"  Outputs: {self.outputs}")
        
        # Show operations
        lines.append("\nOperations:")
        for op in self.operations.values():
            lines.append(f"  {op}")
        
        # Show tensors
        lines.append("\nTensors:")
        for tensor in self.tensors.values():
            lines.append(f"  {tensor}")
        
        return "\n".join(lines)
