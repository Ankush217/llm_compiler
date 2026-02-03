"""
Parameter Accounting
====================

Single source of truth for parameter counting derived from the IR graph.
All other components (solver estimates, emitters, validators) should
cross-check against this to avoid silent divergence.
"""

from __future__ import annotations
from typing import Dict

from ..ir.graph import IRGraph, NodeType


def count_parameters_from_ir(graph: IRGraph) -> int:
    """
    Count trainable parameters directly from the IR graph.

    Currently supports the node types emitted by the decoder-only template.
    Nodes without parameters (add, activation, attention kernels, etc.)
    contribute zero.
    """
    params = 0

    for op in graph.operations.values():
        if op.node_type == NodeType.LINEAR:
            in_features = op.attributes.get("in_features")
            out_features = op.attributes.get("out_features")
            use_bias = op.attributes.get("use_bias", True)
            if in_features and out_features:
                params += in_features * out_features
                if use_bias:
                    params += out_features

        elif op.node_type == NodeType.EMBEDDING:
            vocab_size = op.attributes.get("vocab_size")
            embedding_dim = op.attributes.get("embedding_dim")
            if vocab_size and embedding_dim:
                params += vocab_size * embedding_dim

        elif op.node_type in (NodeType.LAYERNORM, NodeType.RMSNORM):
            normalized_shape = op.attributes.get("normalized_shape")
            if normalized_shape:
                # LayerNorm has weight and bias; RMSNorm has only weight
                if op.node_type == NodeType.LAYERNORM:
                    params += 2 * normalized_shape
                else:
                    params += normalized_shape

        # Multi-head attention parameters are captured by their constituent
        # linear projections in the current IR; nothing to add here.

    return params
