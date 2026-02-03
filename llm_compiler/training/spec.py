from dataclasses import dataclass, field
from typing import List
from .optimizer import OptimizerSpec
from .ir import TrainingIR


@dataclass
class TrainingSpec:
    """
    Declarative training specification.
    Ties together model, dataset, and tokenizer identities plus hyperparameters.
    No execution here—intent only.
    """

    name: str

    model_hash: str
    dataset_hash: str
    tokenizer_hash: str

    objective: str = "causal_lm"

    batch_size: int = 8
    microbatch_size: int = 1
    max_steps: int = 100_000

    optimizer: OptimizerSpec = field(default_factory=lambda: OptimizerSpec(type="adamw", lr=3e-4, betas=(0.9, 0.999)))
    precision: str = "fp16"
    version: str = "v1"

    def validate(self) -> List[str]:
        errors: List[str] = []

        if not self.name:
            errors.append("name is required")

        if not self.model_hash:
            errors.append("model_hash is required")
        if not self.dataset_hash:
            errors.append("dataset_hash is required")
        if not self.tokenizer_hash:
            errors.append("tokenizer_hash is required")

        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.microbatch_size <= 0:
            errors.append("microbatch_size must be positive")
        if self.max_steps <= 0:
            errors.append("max_steps must be positive")
        errors.extend(self.optimizer.validate())

        return errors

    def compile(self) -> TrainingIR:
        errors = self.validate()
        if errors:
            raise ValueError(f"TrainingSpec invalid: {errors}")
        ir = TrainingIR(name=self.name, version=self.version)
        ir.metadata.update({
            "model_hash": self.model_hash,
            "dataset_hash": self.dataset_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "objective": self.objective,
            "batch_size": self.batch_size,
            "microbatch_size": self.microbatch_size,
            "max_steps": self.max_steps,
            "precision": self.precision,
            "optimizer": self.optimizer.to_dict(),
        })
        return ir
