from pathlib import Path
import json
from datetime import datetime

from .validate import validate_training_ir
from .plan import build_training_plan
from .result import ExecutionResult
from .workspace import layout_paths, run_state, ensure_workspace
from .tokenize_real import run_tokenization
from .train_real import run_training_step


def run_training(
    training_ir,
    model_ir,
    dataset_ir,
    tokenizer_ir,
    output_root: Path | str = "runs",
):
    """
    LEVEL 1 execution:
    - real IO (plans, provenance, results)
    - no ML compute
    """
    training_hash = training_ir.fingerprint()
    paths = layout_paths(training_hash, output_root)
    state = run_state(paths)

    if state == "completed":
        return ExecutionResult.load(paths["result"])
    if state == "planned":
        raise RuntimeError(f"Run {training_hash} already planned but not completed")

    # Validate intent
    ok, errors = validate_training_ir(training_ir)
    if not ok:
        return ExecutionResult(
            training_hash=training_hash,
            status="failed",
            artifacts={},
            metrics={},
            logs=errors,
        )

    # Set up workspace
    ensure_workspace(training_hash, output_root)

    # Tokenization stage (idempotent)
    ok_tok, tok_artifacts, tok_err = run_tokenization(
        training_hash, dataset_ir, tokenizer_ir, output_root
    )
    if not ok_tok:
        return tok_err

    # Training checkpoint-only execution
    train_result = run_training_step(
        model_ir_hash=model_ir.fingerprint(),
        tokenization_manifest=tok_artifacts,
        training_ir=training_ir,
        workspace_paths=paths,
    )

    # 1) Build and persist plan (only once)
    plan = build_training_plan(training_ir)
    if not paths["plan"].exists():
        with open(paths["plan"], "w") as f:
            json.dump(plan, f, indent=2)

    # 2) Materialize provenance (hash links, timestamp)
    provenance = {
        "training_hash": training_hash,
        "model_hash": model_ir.fingerprint(),
        "dataset_hash": dataset_ir.fingerprint(),
        "tokenizer_hash": tokenizer_ir.fingerprint(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(paths["root"] / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    # 3) Emit result once
    result = ExecutionResult(
        training_hash=training_hash,
        status="completed",
        artifacts={
            "run_dir": str(paths["root"]),
            "plan": "plan.json",
            "provenance": "provenance.json",
            "tokenization": {
                "manifest": str(paths["token_manifest"].relative_to(paths["root"])),
                "num_shards": tok_artifacts.get("num_shards"),
                "total_tokens": tok_artifacts.get("total_tokens"),
            },
            "checkpoints": train_result.artifacts.get("checkpoints"),
        },
        metrics=train_result.metrics,
        logs=train_result.logs,
    )
    if not paths["result"].exists():
        with open(paths["result"], "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result
