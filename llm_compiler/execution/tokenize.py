from typing import Dict, Any

from ..dataset.ir import DatasetIR
from ..tokenizer.ir import TokenizerIR


def tokenize(dataset_ir: DatasetIR, tokenizer_ir: TokenizerIR) -> Dict[str, Any]:
    """
    Deterministic, side-effect-free tokenization declaration.

    Returns a descriptor of tokenized shards keyed by dataset/tokenizer hashes,
    without performing any IO.
    """
    return {
        "dataset_hash": dataset_ir.fingerprint(),
        "tokenizer_hash": tokenizer_ir.fingerprint(),
        "shards": [
            {
                "id": "shard_0",
                "source": "all",
                "token_count": dataset_ir.metadata.get("target_tokens"),
            }
        ],
        "metadata": {
            "num_shards": 1,
            "note": "tokenization declared only; no IO performed",
        },
    }
