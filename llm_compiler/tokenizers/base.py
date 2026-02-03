"""
Tokenizer base classes
======================

Lightweight implementations to keep tokenizer definition, vocab size,
and corpus statistics bound together. These implementations are minimal
but enforce the invariants needed by the compiler/emitters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Iterable


@dataclass
class TokenizerTrainingStats:
    """Corpus statistics captured during tokenizer training."""
    token_freqs: Dict[str, int]
    total_tokens: int
    unique_tokens: int

    @classmethod
    def from_counts(cls, counts: Dict[str, int]) -> "TokenizerTrainingStats":
        total = sum(counts.values())
        return cls(
            token_freqs=dict(counts),
            total_tokens=total,
            unique_tokens=len(counts),
        )


class BaseTokenizer:
    """Minimal tokenizer contract used by the compiler."""

    def __init__(self, vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.training_stats: TokenizerTrainingStats | None = None

    # --- binding / validation -------------------------------------------------
    def bind_training_stats(self, stats: TokenizerTrainingStats) -> None:
        """Attach corpus statistics and ensure the vocab respects frequencies."""
        self.training_stats = stats
        if stats.unique_tokens < self.vocab_size:
            raise ValueError(
                f"Requested vocab_size={self.vocab_size} exceeds unique tokens in corpus ({stats.unique_tokens})"
            )

    def enforce_vocab_size(self):
        if len(self.id_to_token) != self.vocab_size:
            raise ValueError(
                f"Tokenizer vocab size mismatch: expected {self.vocab_size}, got {len(self.id_to_token)}"
            )

    # --- interface -----------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError

    def train(self, stats: TokenizerTrainingStats):
        """Train tokenizer using provided corpus stats."""
        raise NotImplementedError


class ToyBPETokenizer(BaseTokenizer):
    """
    Minimal stand-in BPE tokenizer.
    Not production-ready but enforces vocab/statistics coupling.
    """

    def train(self, stats: TokenizerTrainingStats):
        self.bind_training_stats(stats)
        # Build vocab by most frequent tokens first
        sorted_tokens = sorted(stats.token_freqs.items(), key=lambda kv: kv[1], reverse=True)
        top_tokens = [tok for tok, _ in sorted_tokens[: self.vocab_size - 2]]
        # Reserve UNK and PAD
        vocab = [self.pad_token, self.unk_token] + top_tokens
        self.id_to_token = vocab[: self.vocab_size]
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.id_to_token)}
        self.enforce_vocab_size()
        return self

    def encode(self, text: str) -> List[int]:
        # Very naive whitespace tokenization for placeholder behavior
        tokens = text.split()
        ids = []
        for tok in tokens:
            ids.append(self.token_to_id.get(tok, self.token_to_id.get(self.unk_token, 1)))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.id_to_token[i] if 0 <= i < len(self.id_to_token) else self.unk_token for i in ids)


class ToyUnigramTokenizer(BaseTokenizer):
    """Simple unigram tokenizer with frequency-aware vocab selection."""

    def train(self, stats: TokenizerTrainingStats):
        self.bind_training_stats(stats)
        sorted_tokens = sorted(stats.token_freqs.items(), key=lambda kv: kv[1], reverse=True)
        top_tokens = [tok for tok, _ in sorted_tokens[: self.vocab_size - 2]]
        vocab = [self.pad_token, self.unk_token] + top_tokens
        self.id_to_token = vocab[: self.vocab_size]
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.id_to_token)}
        self.enforce_vocab_size()
        return self

    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        return [self.token_to_id.get(tok, self.token_to_id.get(self.unk_token, 1)) for tok in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.id_to_token[i] if 0 <= i < len(self.id_to_token) else self.unk_token for i in ids)
