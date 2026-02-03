from typing import Dict, Any, List

from ..training.ir import TrainingIR


def build_training_plan(training_ir: TrainingIR) -> Dict[str, Any]:
    """
    Produce a static training plan from a TrainingIR.
    No IO or execution; purely declarative.
    """
    plan: Dict[str, Any] = {
        "name": training_ir.name,
        "version": training_ir.version,
        "artifacts": [],
        "resources": {},
        "steps": [],
    }

    md = training_ir.metadata

    # Expected artifacts
    plan["artifacts"] = [
        {"type": "checkpoints", "pattern": f"{md.get('model_hash', 'model')}_step_*.pt"},
        {"type": "logs", "pattern": f"train_{training_ir.name}.log"},
        {"type": "metrics", "pattern": f"metrics_{training_ir.name}.json"},
    ]

    # Required resources (declarative hints)
    plan["resources"] = {
        "precision": md.get("precision", "fp16"),
        "optimizer": md.get("optimizer", "adamw"),
        "batch_size": md.get("batch_size"),
        "microbatch_size": md.get("microbatch_size"),
        "max_steps": md.get("max_steps"),
    }

    # Step outline
    plan["steps"] = [
        {
            "name": "validate_links",
            "inputs": ["model_hash", "dataset_hash", "tokenizer_hash"],
            "outputs": [],
        },
        {
            "name": "tokenize_dataset",
            "inputs": ["dataset_ir", "tokenizer_ir"],
            "outputs": ["tokenized_shards"],
        },
        {
            "name": "train_model",
            "inputs": ["model_ir", "tokenized_shards", "training_ir"],
            "outputs": ["checkpoints", "logs", "metrics"],
        },
    ]

    return plan
