import json
from pathlib import Path
from typing import Dict, Any, Tuple
import math
import itertools

from ..dataset.ir import DatasetIR
from ..tokenizer.ir import TokenizerIR
from .workspace import ensure_workspace, layout_paths
from .result import ExecutionResult


def _iter_fake_sequences(total_tokens: int, max_seq_len: int):
    """
    Deterministic synthetic sequence generator to avoid real data access.
    Produces sequences of length up to max_seq_len until total_tokens consumed.
    """
    remaining = total_tokens
    idx = 0
    while remaining > 0:
        length = min(max_seq_len, remaining)
        # Simple deterministic token ids: [idx, idx+1, ...]
        seq = list(range(idx % 1000, idx % 1000 + min(length, 16)))
        # Repeat last token to reach length
        if len(seq) < length:
            seq.extend([seq[-1]] * (length - len(seq)))
        yield seq[:length]
        remaining -= length
        idx += 1


def run_tokenization(
    training_hash: str,
    dataset_ir: DatasetIR,
    tokenizer_ir: TokenizerIR,
    root: Path | str = "runs",
    max_seq_len: int = 4096,
    target_tokens_override: int | None = None,
) -> Tuple[bool, Dict[str, Any], ExecutionResult | None]:
    """
    Tokenization execution (minimal real tokens):
    - resumable: reuses manifest if present
    - writes jsonl shards with input_ids/length
    - deterministic synthetic tokens (no real dataset IO)
    """
    paths = layout_paths(training_hash, root)
    token_manifest = paths["token_manifest"]
    token_shards_dir = paths["token_shards_dir"]

    # Idempotent: if manifest exists, reuse
    if token_manifest.exists():
        with open(token_manifest) as f:
            manifest = json.load(f)
        return True, manifest, None

    # Ensure tokenization dirs
    ensure_workspace(training_hash, root)

    try:
        target_tokens = target_tokens_override or dataset_ir.metadata.get("target_tokens", 0)
        if target_tokens <= 0:
            raise ValueError("target_tokens missing; cannot tokenize")

        # Heuristic shard count
        tokens_per_shard = max_seq_len * 1000  # ~1000 sequences per shard
        num_shards = max(1, math.ceil(target_tokens / tokens_per_shard))

        total_tokens_written = 0
        total_sequences = 0

        token_shards_dir.mkdir(parents=True, exist_ok=True)

        seq_iter = _iter_fake_sequences(target_tokens, max_seq_len)

        for shard_idx in range(num_shards):
            shard_path = token_shards_dir / f"shard_{shard_idx:05d}.jsonl"
            # resumable: if shard exists and non-empty, skip
            if shard_path.exists() and shard_path.stat().st_size > 0:
                continue

            with shard_path.open("w") as out:
                tokens_in_shard = 0
                while tokens_in_shard < tokens_per_shard and total_tokens_written < target_tokens:
                    seq = next(seq_iter)
                    entry = {"input_ids": seq, "length": len(seq)}
                    out.write(json.dumps(entry) + "\n")
                    tokens_in_shard += len(seq)
                    total_tokens_written += len(seq)
                    total_sequences += 1

        manifest = {
            "tokenizer_hash": tokenizer_ir.fingerprint(),
            "dataset_hash": dataset_ir.fingerprint(),
            "num_shards": num_shards,
            "num_sequences": total_sequences,
            "total_tokens": total_tokens_written,
            "max_seq_len": max_seq_len,
            "format": "jsonl",
            "version": 1,
            "shards": [
                {
                    "id": f"shard_{i:05d}",
                    "path": str((token_shards_dir / f"shard_{i:05d}.jsonl").relative_to(paths["root"])),
                }
                for i in range(num_shards)
            ],
        }

        with open(token_manifest, "w") as f:
            json.dump(manifest, f, indent=2)

        return True, manifest, None
    except Exception as e:
        err_result = ExecutionResult(
            training_hash=training_hash,
            status="failed",
            artifacts={},
            metrics={},
            logs=[f"tokenization failed: {e}"],
        )
        return False, {}, err_result
