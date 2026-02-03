"""
Dataset Specification
=====================

Lightweight declaration for datasets used during tokenizer training and
model specification. Couples corpus statistics (token counts/frequencies)
to the tokenizer and ultimately to the architecture spec.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from ..tokenizers.base import TokenizerTrainingStats


@dataclass
class DatasetSpec:
    name: str
    split: str = "train"
    tokenizer_stats: Optional[TokenizerTrainingStats] = None
    path: Optional[str] = None  # optional local path/URI hint
    meta: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.tokenizer_stats:
            data["tokenizer_stats"] = {
                "token_freqs": dict(self.tokenizer_stats.token_freqs),
                "total_tokens": self.tokenizer_stats.total_tokens,
                "unique_tokens": self.tokenizer_stats.unique_tokens,
            }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSpec":
        stats = data.get("tokenizer_stats")
        tokenizer_stats = None
        if stats:
            tokenizer_stats = TokenizerTrainingStats(
                token_freqs=stats.get("token_freqs", {}),
                total_tokens=stats.get("total_tokens", 0),
                unique_tokens=stats.get("unique_tokens", len(stats.get("token_freqs", {}))),
            )
        return cls(
            name=data["name"],
            split=data.get("split", "train"),
            tokenizer_stats=tokenizer_stats,
            path=data.get("path"),
            meta=data.get("meta"),
        )
