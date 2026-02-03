"""
PyTorch Emitter
===============

Emits PyTorch modules that *execute directly from the IR*.
The generated model:
- Parses the frozen IR JSON embedded in the emitted file
- Instantiates nn.Modules only for parameter-carrying IR nodes
- Executes operations in topological order without re-deciding structure
"""

from __future__ import annotations
from typing import Dict, Any, List
import json
import textwrap
from pathlib import Path

from ..ir.graph import IRGraph, NodeType


class PyTorchEmitter:
    """IR-driven PyTorch emitter."""

    def __init__(self, precision: str = "float32"):
        self.precision = precision
        self.torch_dtype_map = {
            "float32": "torch.float32",
            "float16": "torch.float16",
            "bfloat16": "torch.bfloat16",
            "int32": "torch.int32",
            "int64": "torch.int64",
        }

    # ------------------------------------------------------------------ public
    def emit(self, graph: IRGraph, model_name: str, output_dir: Path) -> Dict[str, str]:
        files: Dict[str, str] = {}

        files[f"{model_name}.py"] = self._emit_model_file(graph, model_name)
        files["config.py"] = self._emit_config(graph)
        files["__init__.py"] = self._emit_init_file(model_name)
        files["setup.py"] = self._emit_setup_file(model_name)
        files[f"test_{model_name}.py"] = self._emit_test_file(model_name)
        return files

    # ----------------------------------------------------------------- emitters
    def _emit_model_file(self, graph: IRGraph, model_name: str) -> str:
        """Emit the IR-driven PyTorch model."""
        graph_blob = graph.to_json()
        dtype_literal = self.torch_dtype_map.get(self.precision, "torch.float32")

        model_code = textwrap.dedent(
            f'''
            import json
            from typing import Dict, Any, Optional
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            from .config import ModelConfig

            IR_JSON = r"""{graph_blob}"""
            IR_DEF = json.loads(IR_JSON)


            # ------------------------------------------------------------- helpers
            def _topological_sort(ir: Dict[str, Any]):
                ops = ir["operations"]
                tensors = ir["tensors"]
                indegree = {{name: 0 for name in ops}}
                for op_name, op in ops.items():
                    for inp in op["inputs"]:
                        producer = tensors.get(inp, {{}}).get("node")
                        if producer and producer in indegree:
                            indegree[op_name] += 1
                ready = [name for name, deg in indegree.items() if deg == 0]
                order = []
                while ready:
                    current = ready.pop(0)
                    order.append(current)
                    for out in ops[current]["outputs"]:
                        for consumer in tensors.get(out, {{}}).get("consumers", []):
                            if consumer in indegree:
                                indegree[consumer] -= 1
                                if indegree[consumer] == 0:
                                    ready.append(consumer)
                if len(order) != len(ops):
                    # Fallback to insertion order if a cycle slipped in
                    return list(ops.keys())
                return order


            class RMSNorm(nn.Module):
                def __init__(self, dim: int, eps: float = 1e-6):
                    super().__init__()
                    self.eps = eps
                    self.weight = nn.Parameter(torch.ones(dim))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    norm_x = x.pow(2).mean(-1, keepdim=True)
                    x_normed = x * torch.rsqrt(norm_x + self.eps)
                    return self.weight * x_normed


            class AttentionOp(nn.Module):
                def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, dropout: float = 0.0):
                    super().__init__()
                    self.num_heads = num_heads
                    self.num_kv_heads = num_kv_heads
                    self.head_dim = head_dim
                    self.dropout = dropout
                    self.num_heads_per_kv = max(1, num_heads // num_kv_heads)

                def forward(self, q, k, v, attention_mask=None):
                    bsz, seq_len, _ = q.shape
                    q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

                    if self.num_kv_heads != self.num_heads:
                        k = k.repeat_interleave(self.num_heads_per_kv, dim=1)
                        v = v.repeat_interleave(self.num_heads_per_kv, dim=1)

                    attn = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attention_mask, dropout_p=self.dropout
                    )
                    attn = attn.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
                    return attn


            def _apply_rope(x: torch.Tensor, theta: float = 10000.0):
                if x.size(-1) % 2 != 0:
                    return x
                half = x.size(-1) // 2
                freqs = torch.arange(half, device=x.device, dtype=x.dtype)
                freqs = theta ** (-freqs / half)
                positions = torch.arange(x.size(1), device=x.device, dtype=x.dtype)
                angles = torch.einsum("i,j->ij", positions, freqs)
                sin, cos = angles.sin(), angles.cos()
                x1, x2 = x[..., :half], x[..., half:]
                rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
                return rotated.reshape_as(x)


            def _activation(kind: str, tensor: torch.Tensor):
                if kind == "relu":
                    return F.relu(tensor)
                if kind == "gelu":
                    return F.gelu(tensor)
                if kind == "silu":
                    return F.silu(tensor)
                if kind == "swiglu":
                    # Caller supplies gate/up split; we only keep hook
                    return tensor
                raise ValueError(f"Unsupported activation: {{kind}}")


            # -------------------------------------------------------------- model
            class {model_name}(nn.Module):
                def __init__(self, config: Optional[ModelConfig] = None):
                    super().__init__()
                    self.ir = IR_DEF
                    self.order = _topological_sort(IR_DEF)
                    self.config = config or ModelConfig()
                    self.modules_by_op = nn.ModuleDict()

                    for name, op in self.ir["operations"].items():
                        op_type = op["type"]
                        attrs = op.get("attributes", {{}})
                        if op_type == "embedding":
                            self.modules_by_op[name] = nn.Embedding(
                                attrs["vocab_size"], attrs["embedding_dim"]
                            )
                        elif op_type == "linear":
                            self.modules_by_op[name] = nn.Linear(
                                attrs["in_features"],
                                attrs["out_features"],
                                bias=attrs.get("use_bias", True),
                            )
                        elif op_type == "rmsnorm":
                            self.modules_by_op[name] = RMSNorm(
                                attrs["normalized_shape"], eps=attrs.get("eps", 1e-6)
                            )
                        elif op_type == "layernorm":
                            self.modules_by_op[name] = nn.LayerNorm(
                                attrs["normalized_shape"], eps=attrs.get("eps", 1e-5)
                            )
                        elif op_type == "multi_head_attention":
                            self.modules_by_op[name] = AttentionOp(
                                attrs["num_heads"],
                                attrs.get("num_kv_heads", attrs["num_heads"]),
                                attrs["head_dim"],
                                dropout=self.config.attention_dropout,
                            )
                        # Parameter-free ops are executed directly in forward

                    # Handle tied weights declared in the IR
                    for name, op in self.ir["operations"].items():
                        attrs = op.get("attributes", {{}})
                        tie_target = attrs.get("tie_weight")
                        if tie_target and name in self.modules_by_op:
                            target_module = tie_target.split(".")[0]
                            if target_module in self.modules_by_op:
                                self.modules_by_op[name].weight = self.modules_by_op[target_module].weight

                    # Move to configured dtype
                    self.to(dtype={dtype_literal})

                def forward(self, **inputs):
                    values: Dict[str, torch.Tensor] = {{}}

                    # Bind graph inputs
                    for required in self.ir["inputs"]:
                        if required not in inputs:
                            raise ValueError(f"Missing required input '{{required}}'")
                        values[required] = inputs[required]

                    attention_mask = inputs.get("attention_mask")

                    for op_name in self.order:
                        op = self.ir["operations"][op_name]
                        op_type = op["type"]
                        attrs = op.get("attributes", {{}})
                        args = [values[i] for i in op.get("inputs", [])]

                        if op_type == "embedding":
                            out = self.modules_by_op[op_name](args[0])
                        elif op_type == "linear":
                            out = self.modules_by_op[op_name](args[0])
                        elif op_type == "rmsnorm" or op_type == "layernorm":
                            out = self.modules_by_op[op_name](args[0])
                        elif op_type == "add":
                            out = args[0] + args[1]
                        elif op_type == "mul":
                            out = args[0] * args[1]
                        elif op_type == "activation":
                            out = _activation(attrs.get("activation", "silu"), args[0])
                        elif op_type == "swiglu":
                            out = F.silu(args[0]) * args[1]
                        elif op_type == "multi_head_attention":
                            out = self.modules_by_op[op_name](args[0], args[1], args[2], attention_mask)
                        elif op_type == "rope":
                            out = _apply_rope(args[0], theta=attrs.get("theta", 10000.0))
                        elif op_type == "softmax":
                            out = F.softmax(args[0], dim=attrs.get("dim", -1))
                        else:
                            raise RuntimeError(f"Unsupported IR op type: {{op_type}}")

                        # Assume single output per op
                        out_name = op["outputs"][0]
                        values[out_name] = out

                    # Collect graph outputs
                    outputs = {{name: values[name] for name in self.ir["outputs"]}}
                    if len(outputs) == 1:
                        return next(iter(outputs.values()))
                    return outputs
            '''
        )

        return model_code

    def _derive_config_defaults(self, graph: IRGraph) -> Dict[str, Any]:
        dims: Dict[str, Any] = {}
        # Use first embedding as authoritative vocab/hidden sizes
        for op in graph.operations.values():
            if op.node_type == NodeType.EMBEDDING:
                dims["vocab_size"] = op.attributes.get("vocab_size", 50000)
                dims["hidden_size"] = op.attributes.get("embedding_dim", 4096)
                break

        # Heads / head_dim from first attention op
        for op in graph.operations.values():
            if op.node_type == NodeType.MULTI_HEAD_ATTENTION:
                dims["num_attention_heads"] = op.attributes.get("num_heads", 32)
                dims["num_key_value_heads"] = op.attributes.get("num_kv_heads", dims["num_attention_heads"])
                dims["head_dim"] = op.attributes.get("head_dim", 128)
                break

        # Intermediate size from first gate/up projection
        for op in graph.operations.values():
            if op.node_type == NodeType.LINEAR and "up_proj" in op.name:
                dims["intermediate_size"] = op.attributes.get("out_features", 11008)
                break

        # Layers
        dims["num_hidden_layers"] = sum(1 for op in graph.operations.values() if op.node_type == NodeType.MULTI_HEAD_ATTENTION)

        # Context length from input tensor shapes
        for tensor in graph.tensors.values():
            if tensor.name in graph.inputs and tensor.shape:
                if len(tensor.shape) >= 2 and tensor.shape[1] not in (-1, None):
                    dims["context_length"] = tensor.shape[1]
                    break
        dims.setdefault("context_length", 8192)

        return dims

    def _emit_config(self, graph: IRGraph) -> str:
        dims = self._derive_config_defaults(graph)
        return textwrap.dedent(
            f"""
            from dataclasses import dataclass
            from typing import Optional


            @dataclass
            class ModelConfig:
                vocab_size: int = {dims.get('vocab_size', 50000)}
                hidden_size: int = {dims.get('hidden_size', 4096)}
                intermediate_size: int = {dims.get('intermediate_size', 11008)}
                num_hidden_layers: int = {dims.get('num_hidden_layers', 32)}
                num_attention_heads: int = {dims.get('num_attention_heads', 32)}
                num_key_value_heads: int = {dims.get('num_key_value_heads', dims.get('num_attention_heads', 32))}
                head_dim: int = {dims.get('head_dim', 128)}
                max_position_embeddings: int = {dims.get('context_length', 8192)}

                attention_dropout: float = 0.0
                hidden_dropout: float = 0.0
                torch_dtype: str = "{self.precision}"

                def to_dict(self):
                    return self.__dict__.copy()
            """
        )

    def _emit_init_file(self, model_name: str) -> str:
        return textwrap.dedent(
            f"""
            from .config import ModelConfig
            from .{model_name} import {model_name}

            __all__ = [
                "ModelConfig",
                "{model_name}",
            ]

            __version__ = "1.0.0"
            """
        )

    def _emit_setup_file(self, model_name: str) -> str:
        return textwrap.dedent(
            f"""
            from setuptools import setup, find_packages

            setup(
                name="{model_name.lower()}",
                version="1.0.0",
                author="LLM Compiler",
                description="Generated LLM model",
                packages=find_packages(),
                python_requires=">=3.8",
                install_requires=[
                    "torch>=2.0.0",
                ],
            )
            """
        )

    def _emit_test_file(self, model_name: str) -> str:
        return textwrap.dedent(
            f"""
            import torch
            from .config import ModelConfig
            from .{model_name} import {model_name}


            def test_model():
                config = ModelConfig()
                model = {model_name}(config)
                batch_size = 2
                seq_len = min(16, config.max_position_embeddings)

                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
                outputs = model(input_ids=input_ids)
                if isinstance(outputs, dict):
                    logits = list(outputs.values())[0]
                else:
                    logits = outputs
                assert logits.shape[:2] == (batch_size, seq_len)
                print("Model forward succeeded; output shape", logits.shape)


            if __name__ == "__main__":
                test_model()
            """
        )
