from pathlib import Path
from typing import Dict, Any

from .spec import DatasetSpec
from .ir import DatasetIR


class DatasetCompiler:
    """
    Compiles DatasetSpec → DatasetIR.
    No IO, no crawling. Deterministic intent-only.
    """

    def compile(self, spec: DatasetSpec) -> DatasetIR:
        errors = spec.validate()
        if errors:
            raise ValueError(f"DatasetSpec invalid: {errors}")

        ir = DatasetIR(
            name=spec.name,
            version=spec.version,
        )

        # Declare sources (symbolic)
        for domain in spec.domains:
            ir.add_node("source", domain=domain)

        # Declare filters
        ir.add_node(
            "length_filter",
            min_length=spec.min_length,
            max_length=spec.max_length,
        )

        if spec.deduplicate:
            ir.add_node("deduplicate", method="minhash")

        if spec.remove_boilerplate:
            ir.add_node("boilerplate_filter")

        # Declare language mix
        ir.add_node(
            "language_mix",
            ratios=spec.languages
        )

        # Declare size intent
        ir.metadata["target_tokens"] = spec.target_tokens

        return ir
