from dataclasses import dataclass, field
from typing import Dict, List, Literal
import json
from pathlib import Path


@dataclass
class ExecutionResult:
    """
    Container for execution outcomes, separate from intent.
    """
    training_hash: str
    status: Literal["planned", "completed", "failed"] = "planned"
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, any]:
        return {
            "training_hash": self.training_hash,
            "status": self.status,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "logs": self.logs,
        }

    @classmethod
    def load(cls, path: Path) -> "ExecutionResult":
        data = json.loads(path.read_text())
        return cls(
            training_hash=data["training_hash"],
            status=data.get("status", "completed"),
            artifacts=data.get("artifacts", {}),
            metrics=data.get("metrics", {}),
            logs=data.get("logs", []),
        )
