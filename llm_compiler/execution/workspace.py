from pathlib import Path
from typing import Dict


def layout_paths(training_hash: str, root: Path | str = "runs") -> Dict[str, Path]:
    """
    Return canonical workspace paths for a training run (no IO).
    
    Structure:
    runs/<training_hash>/
        plan.json
        dataset/
        tokenizer/
        model/
        artifacts/
        logs/
        result.json
    """
    root = Path(root)
    run_dir = root / training_hash
    return {
        "root": run_dir,
        "plan": run_dir / "plan.json",
        "dataset_dir": run_dir / "dataset",
        "tokenizer_dir": run_dir / "tokenizer",
        "model_dir": run_dir / "model",
        "artifacts_dir": run_dir / "artifacts",
        "logs_dir": run_dir / "logs",
        "result": run_dir / "result.json",
        "tokenization_dir": run_dir / "tokenization",
        "token_manifest": run_dir / "tokenization" / "manifest.json",
        "token_shards_dir": run_dir / "tokenization" / "shards",
        "training_dir": run_dir / "training",
        "checkpoints_dir": run_dir / "training" / "checkpoints",
        "metrics_dir": run_dir / "training" / "metrics",
        "train_logs_dir": run_dir / "training" / "logs",
    }


def ensure_workspace(training_hash: str, root: Path | str = "runs") -> Dict[str, Path]:
    """
    Create the canonical workspace directories (idempotent).
    Does not write plan/result; only ensures directories exist.
    """
    paths = layout_paths(training_hash, root)
    for key in [
        "root",
        "dataset_dir",
        "tokenizer_dir",
        "model_dir",
        "artifacts_dir",
        "logs_dir",
        "tokenization_dir",
        "token_shards_dir",
        "training_dir",
        "checkpoints_dir",
        "metrics_dir",
        "train_logs_dir",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def run_state(paths: Dict[str, Path]) -> str:
    """
    Determine execution state from filesystem.
    """
    result = paths["result"]
    plan = paths["plan"]

    if result.exists():
        return "completed"
    if plan.exists():
        return "planned"
    return "empty"
