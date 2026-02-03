from typing import Dict, Any
import math

from ..training.ir import TrainingIR


def train(model_ir_hash: str, tokenized_descriptor: Dict[str, Any], training_ir: TrainingIR) -> Dict[str, Any]:
    """
    Declarative training planning (still no gradients).
    Reads tokenization manifest to derive steps/epochs expectations.
    """
    total_tokens = tokenized_descriptor.get("total_tokens", 0)
    num_sequences = tokenized_descriptor.get("num_sequences", 0)
    max_seq_len = tokenized_descriptor.get("max_seq_len", 0)

    md = training_ir.metadata
    batch_size = md.get("batch_size", 8)
    max_steps = md.get("max_steps", 100_000)

    avg_seq_len = total_tokens / num_sequences if num_sequences else max_seq_len
    tokens_per_step = batch_size * avg_seq_len
    steps_per_epoch = math.ceil(total_tokens / tokens_per_step) if tokens_per_step else 0

    return {
        "model_ir_hash": model_ir_hash,
        "tokenized_dataset_hash": tokenized_descriptor.get("dataset_hash"),
        "tokenizer_hash": tokenized_descriptor.get("tokenizer_hash"),
        "training_hash": training_ir.fingerprint(),
        "derived": {
            "total_tokens": total_tokens,
            "num_sequences": num_sequences,
            "tokens_per_step": tokens_per_step,
            "steps_per_epoch": steps_per_epoch,
            "planned_steps": max_steps,
        },
        "expected_outputs": {
            "checkpoints_pattern": f"{model_ir_hash}_step_*.pt",
            "logs": f"train_{training_ir.name}.log",
            "metrics": f"metrics_{training_ir.name}.json",
        },
        "note": "training declaration only; gradients not executed",
    }
