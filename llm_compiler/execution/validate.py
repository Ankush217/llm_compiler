from typing import Iterable, Tuple, List, Optional

from ..training.ir import TrainingIR


def validate_training_ir(
    training_ir: TrainingIR,
    known_hashes: Optional[Iterable[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Stateless validation for a TrainingIR.

    - Verifies required hashes are present.
    - Optionally checks hashes exist in a provided collection (no IO).
    - Ensures hash fields are non-empty strings.
    """
    errors: List[str] = []

    # Extract expected linkage from metadata
    model_hash = training_ir.metadata.get("model_hash")
    dataset_hash = training_ir.metadata.get("dataset_hash")
    tokenizer_hash = training_ir.metadata.get("tokenizer_hash")

    for field_name, value in [
        ("model_hash", model_hash),
        ("dataset_hash", dataset_hash),
        ("tokenizer_hash", tokenizer_hash),
    ]:
        if not value or not isinstance(value, str):
            errors.append(f"{field_name} missing or not a string")

    if known_hashes is not None:
        known = set(known_hashes)
        if model_hash and model_hash not in known:
            errors.append(f"model_hash {model_hash} not found in known_hashes")
        if dataset_hash and dataset_hash not in known:
            errors.append(f"dataset_hash {dataset_hash} not found in known_hashes")
        if tokenizer_hash and tokenizer_hash not in known:
            errors.append(f"tokenizer_hash {tokenizer_hash} not found in known_hashes")

    return len(errors) == 0, errors
