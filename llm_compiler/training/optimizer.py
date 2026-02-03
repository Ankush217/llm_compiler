from dataclasses import dataclass, asdict
from typing import Literal, Tuple, Dict, Any
import json
import hashlib


@dataclass
class OptimizerSpec:
    type: Literal["adamw", "sgd"]
    lr: float
    betas: Tuple[float, float] | None = None
    weight_decay: float = 0.0

    def validate(self):
        errors = []
        if self.lr <= 0:
            errors.append("optimizer lr must be positive")
        if self.type == "adamw" and self.betas is not None:
            if len(self.betas) != 2:
                errors.append("betas must be a tuple of length 2")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class OptimizerIR:
    def __init__(self, spec: OptimizerSpec):
        self.spec = spec

    def to_dict(self) -> Dict[str, Any]:
        return self.spec.to_dict()

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()
