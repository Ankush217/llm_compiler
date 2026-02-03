
            import json
            from typing import Dict, Any, Optional
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            from .config import ModelConfig

            IR_JSON = r"""{
  "name": "decoder_only_v1",
  "tensors": {
    "input_ids": {
      "name": "input_ids",
      "shape": [
        -1,
        16
      ],
      "dtype": "int32",
      "node": null,
      "consumers": [
        "token_embeddings"
      ]
    },
    "token_embeddings_out_1": {
      "name": "token_embeddings_out_1",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "token_embeddings",
      "consumers": [
        "rope_positional"
      ]
    },
    "rope_positional_out_2": {
      "name": "rope_positional_out_2",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "rope_positional",
      "consumers": [
        "input_norm"
      ]
    },
    "input_norm_out_3": {
      "name": "input_norm_out_3",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "input_norm",
      "consumers": [
        "layer_0_attn_norm",
        "layer_0_attn_residual"
      ]
    },
    "layer_0_attn_norm_out_4": {
      "name": "layer_0_attn_norm_out_4",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "layer_0_attn_norm",
      "consumers": [
        "layer_0_q_proj",
        "layer_0_k_proj",
        "layer_0_v_proj"
      ]
    },
    "layer_0_q_proj_out_5": {
      "name": "layer_0_q_proj_out_5",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "layer_0_q_proj",
      "consumers": [
        "layer_0_attn"
      ]
    },
    "layer_0_k_proj_out_6": {
      "name": "layer_0_k_proj_out_6",
      "shape": [
        -1,
        -1,
        8
      ],
      "dtype": "float32",
      "node": "layer_0_k_proj",
      "consumers": [
        "layer_0_attn"
      ]
    },
    "layer_0_v_proj_out_7": {
      "name": "layer_0_v_proj_out_7",
      "shape": [
        -1,
        -1,
        8
      ],
      "dtype": "float32",
      "node": "layer_0_v_proj",
      "consumers": [
        "layer_0_attn"
      ]
    },
    "layer_0_attn_out_8": {
      "name": "layer_0_attn_out_8",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "layer_0_attn",
      "consumers": [
        "layer_0_attn_out_proj"
      ]
    },
    "layer_0_attn_out_proj_out_9": {
      "name": "layer_0_attn_out_proj_out_9",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "layer_0_attn_out_proj",
      "consumers": [
        "layer_0_attn_residual"
      ]
    },
    "layer_0_attn_residual_out_10": {
      "name": "layer_0_attn_residual_out_10",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "layer_0_attn_residual",
      "consumers": [
        "layer_0_mlp_norm",
        "layer_0_mlp_residual"
      ]
    },
    "layer_0_mlp_norm_out_11": {
      "name": "layer_0_mlp_norm_out_11",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "layer_0_mlp_norm",
      "consumers": [
        "layer_0_fc1"
      ]
    },
    "layer_0_fc1_out_12": {
      "name": "layer_0_fc1_out_12",
      "shape": [
        -1,
        -1,
        32
      ],
      "dtype": "float32",
      "node": "layer_0_fc1",
      "consumers": [
        "layer_0_activation"
      ]
    },
    "layer_0_activation_out_13": {
      "name": "layer_0_activation_out_13",
      "shape": [
        -1,
        -1,
        32
      ],
      "dtype": "float32",
      "node": "layer_0_activation",
      "consumers": [
        "layer_0_down_proj"
      ]
    },
    "layer_0_down_proj_out_14": {
      "name": "layer_0_down_proj_out_14",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "layer_0_down_proj",
      "consumers": [
        "layer_0_mlp_residual"
      ]
    },
    "layer_0_mlp_residual_out_15": {
      "name": "layer_0_mlp_residual_out_15",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "layer_0_mlp_residual",
      "consumers": [
        "output_norm"
      ]
    },
    "output_norm_out_16": {
      "name": "output_norm_out_16",
      "shape": [
        -1,
        -1,
        16
      ],
      "dtype": "float32",
      "node": "output_norm",
      "consumers": [
        "output_projection"
      ]
    },
    "output_projection_out_17": {
      "name": "output_projection_out_17",
      "shape": [
        -1,
        -1,
        32
      ],
      "dtype": "float32",
      "node": "output_projection",
      "consumers": []
    }
  },
  "operations": {
    "token_embeddings": {
      "name": "token_embeddings",
      "type": "embedding",
      "inputs": [
        "input_ids"
      ],
      "outputs": [
        "token_embeddings_out_1"
      ],
      "attributes": {
        "vocab_size": 32,
        "embedding_dim": 16
      }
    },
    "rope_positional": {
      "name": "rope_positional",
      "type": "rope",
      "inputs": [
        "token_embeddings_out_1"
      ],
      "outputs": [
        "rope_positional_out_2"
      ],
      "attributes": {
        "dim": 8,
        "theta": 10000.0
      }
    },
    "input_norm": {
      "name": "input_norm",
      "type": "rmsnorm",
      "inputs": [
        "rope_positional_out_2"
      ],
      "outputs": [
        "input_norm_out_3"
      ],
      "attributes": {
        "normalized_shape": 16,
        "eps": 1e-06
      }
    },
    "layer_0_attn_norm": {
      "name": "layer_0_attn_norm",
      "type": "rmsnorm",
      "inputs": [
        "input_norm_out_3"
      ],
      "outputs": [
        "layer_0_attn_norm_out_4"
      ],
      "attributes": {
        "normalized_shape": 16,
        "eps": 1e-06
      }
    },
    "layer_0_q_proj": {
      "name": "layer_0_q_proj",
      "type": "linear",
      "inputs": [
        "layer_0_attn_norm_out_4"
      ],
      "outputs": [
        "layer_0_q_proj_out_5"
      ],
      "attributes": {
        "in_features": 16,
        "out_features": 16,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_k_proj": {
      "name": "layer_0_k_proj",
      "type": "linear",
      "inputs": [
        "layer_0_attn_norm_out_4"
      ],
      "outputs": [
        "layer_0_k_proj_out_6"
      ],
      "attributes": {
        "in_features": 16,
        "out_features": 8,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_v_proj": {
      "name": "layer_0_v_proj",
      "type": "linear",
      "inputs": [
        "layer_0_attn_norm_out_4"
      ],
      "outputs": [
        "layer_0_v_proj_out_7"
      ],
      "attributes": {
        "in_features": 16,
        "out_features": 8,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_attn": {
      "name": "layer_0_attn",
      "type": "multi_head_attention",
      "inputs": [
        "layer_0_q_proj_out_5",
        "layer_0_k_proj_out_6",
        "layer_0_v_proj_out_7"
      ],
      "outputs": [
        "layer_0_attn_out_8"
      ],
      "attributes": {
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 8,
        "attention_type": "gqa",
        "use_alibi": false
      }
    },
    "layer_0_attn_out_proj": {
      "name": "layer_0_attn_out_proj",
      "type": "linear",
      "inputs": [
        "layer_0_attn_out_8"
      ],
      "outputs": [
        "layer_0_attn_out_proj_out_9"
      ],
      "attributes": {
        "in_features": 16,
        "out_features": 16,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_attn_residual": {
      "name": "layer_0_attn_residual",
      "type": "add",
      "inputs": [
        "input_norm_out_3",
        "layer_0_attn_out_proj_out_9"
      ],
      "outputs": [
        "layer_0_attn_residual_out_10"
      ],
      "attributes": {}
    },
    "layer_0_mlp_norm": {
      "name": "layer_0_mlp_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_0_attn_residual_out_10"
      ],
      "outputs": [
        "layer_0_mlp_norm_out_11"
      ],
      "attributes": {
        "normalized_shape": 16,
        "eps": 1e-06
      }
    },
    "layer_0_fc1": {
      "name": "layer_0_fc1",
      "type": "linear",
      "inputs": [
        "layer_0_mlp_norm_out_11"
      ],
      "outputs": [
        "layer_0_fc1_out_12"
      ],
      "attributes": {
        "in_features": 16,
        "out_features": 32,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_activation": {
      "name": "layer_0_activation",
      "type": "activation",
      "inputs": [
        "layer_0_fc1_out_12"
      ],
      "outputs": [
        "layer_0_activation_out_13"
      ],
      "attributes": {
        "activation": "relu"
      }
    },
    "layer_0_down_proj": {
      "name": "layer_0_down_proj",
      "type": "linear",
      "inputs": [
        "layer_0_activation_out_13"
      ],
      "outputs": [
        "layer_0_down_proj_out_14"
      ],
      "attributes": {
        "in_features": 32,
        "out_features": 16,
        "use_bias": true,
        "tie_weight": null
      }
    },
    "layer_0_mlp_residual": {
      "name": "layer_0_mlp_residual",
      "type": "add",
      "inputs": [
        "layer_0_attn_residual_out_10",
        "layer_0_down_proj_out_14"
      ],
      "outputs": [
        "layer_0_mlp_residual_out_15"
      ],
      "attributes": {}
    },
    "output_norm": {
      "name": "output_norm",
      "type": "rmsnorm",
      "inputs": [
        "layer_0_mlp_residual_out_15"
      ],
      "outputs": [
        "output_norm_out_16"
      ],
      "attributes": {
        "normalized_shape": 16,
        "eps": 1e-06
      }
    },
    "output_projection": {
      "name": "output_projection",
      "type": "linear",
      "inputs": [
        "output_norm_out_16"
      ],
      "outputs": [
        "output_projection_out_17"
      ],
      "attributes": {
        "in_features": 16,
        "out_features": 32,
        "use_bias": false,
        "tie_weight": "token_embeddings.weight"
      }
    }
  },
  "inputs": [
    "input_ids"
  ],
  "outputs": [
    "output_projection_out_17"
  ]
}"""
            IR_DEF = json.loads(IR_JSON)


            # ------------------------------------------------------------- helpers
            def _topological_sort(ir: Dict[str, Any]):
                ops = ir["operations"]
                tensors = ir["tensors"]
                indegree = {name: 0 for name in ops}
                for op_name, op in ops.items():
                    for inp in op["inputs"]:
                        producer = tensors.get(inp, {}).get("node")
                        if producer and producer in indegree:
                            indegree[op_name] += 1
                ready = [name for name, deg in indegree.items() if deg == 0]
                order = []
                while ready:
                    current = ready.pop(0)
                    order.append(current)
                    for out in ops[current]["outputs"]:
                        for consumer in tensors.get(out, {}).get("consumers", []):
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
                raise ValueError(f"Unsupported activation: {kind}")


            # -------------------------------------------------------------- model
            class llmc-decoder-only-0m-relu(nn.Module):
                def __init__(self, config: Optional[ModelConfig] = None):
                    super().__init__()
                    self.ir = IR_DEF
                    self.order = _topological_sort(IR_DEF)
                    self.config = config or ModelConfig()
                    self.modules_by_op = nn.ModuleDict()

                    for name, op in self.ir["operations"].items():
                        op_type = op["type"]
                        attrs = op.get("attributes", {})
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
                        attrs = op.get("attributes", {})
                        tie_target = attrs.get("tie_weight")
                        if tie_target and name in self.modules_by_op:
                            target_module = tie_target.split(".")[0]
                            if target_module in self.modules_by_op:
                                self.modules_by_op[name].weight = self.modules_by_op[target_module].weight

                    # Move to configured dtype
                    self.to(dtype=torch.float32)

                def forward(self, **inputs):
                    values: Dict[str, torch.Tensor] = {}

                    # Bind graph inputs
                    for required in self.ir["inputs"]:
                        if required not in inputs:
                            raise ValueError(f"Missing required input '{required}'")
                        values[required] = inputs[required]

                    attention_mask = inputs.get("attention_mask")

                    for op_name in self.order:
                        op = self.ir["operations"][op_name]
                        op_type = op["type"]
                        attrs = op.get("attributes", {})
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
                            raise RuntimeError(f"Unsupported IR op type: {op_type}")

                        # Assume single output per op
                        out_name = op["outputs"][0]
                        values[out_name] = out

                    # Collect graph outputs
                    outputs = {name: values[name] for name in self.ir["outputs"]}
                    if len(outputs) == 1:
                        return next(iter(outputs.values()))
                    return outputs
