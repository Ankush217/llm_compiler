from dataclasses import dataclass, field
from typing import Dict, Any
import json
import hashlib


@dataclass
class TrainingIR:
    """
    Training Intermediate Representation.
    Canonical, reproducible definition of a training run intent.
    """

    name: str
    version: str

    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "metadata_keys": list(self.metadata.keys()),
        }

    def fingerprint(self) -> str:
        blob = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "metadata": self.metadata,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()
