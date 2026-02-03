import json
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import time

from .workspace import layout_paths, ensure_workspace
from .result import ExecutionResult


class DummyForwardModel(nn.Module):
    """Fallback tiny model if a real generated model is unavailable."""

    def __init__(self, vocab: int, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = x.mean(dim=1)
        return self.head(x)


def _load_model(model_ir_hash: str, device: torch.device, vocab: int) -> nn.Module:
    # TODO: optionally import generated model; for now use dummy
    model = DummyForwardModel(vocab)
    return model.to(device)


def run_training_step(
    model_ir_hash: str,
    tokenization_manifest: Dict[str, Any],
    training_ir,
    workspace_paths: Dict[str, Path],
    device: str = "cpu",
    checkpoint_interval: int = 10,
) -> ExecutionResult:
    """
    Execute checkpoint-only training:
    - runs forward passes over token shards
    - saves checkpoints and metrics
    - no backward/optimizer
    """
    device = torch.device(device)
    ensure_workspace(training_ir.fingerprint(), workspace_paths["root"].parent)

    ckpt_dir = workspace_paths["checkpoints_dir"]
    metrics_dir = workspace_paths["metrics_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Derive vocab from manifest tokens if available
    max_token_id = 0
    for shard in tokenization_manifest.get("shards", []):
        shard_path = workspace_paths["root"] / shard["path"]
        if not shard_path.exists():
            continue
        with shard_path.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec["input_ids"]:
                    max_token_id = max(max_token_id, max(rec["input_ids"]))
    vocab = max_token_id + 1 if max_token_id else 1000

    model = _load_model(model_ir_hash, device, vocab)
    optimizer_spec = training_ir.metadata.get(
        "optimizer",
        {"type": "adamw", "lr": 3e-4, "betas": (0.9, 0.999), "weight_decay": 0.0},
    )
    opt = None
    if optimizer_spec["type"] == "adamw":
        betas = optimizer_spec.get("betas") or (0.9, 0.999)
        opt = optim.AdamW(model.parameters(), lr=optimizer_spec["lr"], betas=betas, weight_decay=optimizer_spec.get("weight_decay", 0.0))
    elif optimizer_spec["type"] == "sgd":
        opt = optim.SGD(model.parameters(), lr=optimizer_spec["lr"], weight_decay=optimizer_spec.get("weight_decay", 0.0))

    loss_fn = nn.CrossEntropyLoss()

    step = 0
    total_sequences = 0
    total_tokens = 0
    start_time = time.time()

    for shard in tokenization_manifest.get("shards", []):
        shard_path = workspace_paths["root"] / shard["path"]
        if not shard_path.exists():
            continue
        with shard_path.open() as f:
            for line in f:
                rec = json.loads(line)
                input_ids = torch.tensor(rec["input_ids"], device=device).unsqueeze(0)
                # simple next-token objective using last token as target
                labels = input_ids[:, -1]
                logits = model(input_ids)
                loss = loss_fn(logits, labels)

                loss.backward()
                if opt is not None:
                    opt.step()
                    opt.zero_grad()

                step += 1
                total_sequences += 1
                total_tokens += len(rec["input_ids"])

                if step % checkpoint_interval == 0:
                    ckpt_file = ckpt_dir / f"step_{step:06d}.pt"
                    torch.save(model.state_dict(), ckpt_file)
                    metrics_file = workspace_paths["metrics_dir"] / f"step_{step:06d}.json"
                    elapsed = time.time() - start_time
                    metrics_file.write_text(json.dumps({
                        "step": step,
                        "sequences": total_sequences,
                        "tokens": total_tokens,
                        "loss": loss.item(),
                        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
                        "steps_per_sec": step / elapsed if elapsed > 0 else 0.0,
                    }))

    # final checkpoint/metrics
    ckpt_file = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(model.state_dict(), ckpt_file)
    metrics_file = workspace_paths["metrics_dir"] / f"step_{step:06d}.json"
    elapsed = time.time() - start_time
    metrics_file.write_text(json.dumps({
        "step": step,
        "sequences": total_sequences,
        "tokens": total_tokens,
        "loss": float(loss.item()) if 'loss' in locals() else 0.0,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        "steps_per_sec": step / elapsed if elapsed > 0 else 0.0,
    }))

    return ExecutionResult(
        training_hash=training_ir.fingerprint(),
        status="completed",
        artifacts={
            "checkpoints": str(ckpt_dir.relative_to(workspace_paths["root"])),
        },
        metrics={
            "steps_completed": step,
            "sequences": total_sequences,
            "tokens": total_tokens,
        },
        logs=["Checkpoint-only training run completed"],
    )
