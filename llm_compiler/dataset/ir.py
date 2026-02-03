from dataclasses import dataclass, field
from typing import Dict, List, Any
import json
import hashlib


@dataclass
class DatasetIR:
    """
    Dataset Intermediate Representation.
    Canonical, reproducible definition of a dataset pipeline.
    """

    name: str
    version: str

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node_type: str, **attrs):
        self.nodes.append({
            "type": node_type,
            "attrs": attrs
        })

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "num_nodes": len(self.nodes),
            "nodes": [n["type"] for n in self.nodes],
        }

    def fingerprint(self) -> str:
        """Deterministic hash of the dataset IR."""
        blob = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "nodes": self.nodes,
                "metadata": self.metadata,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()
