from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TokenizerSpec:
    """
    Declarative tokenizer specification.
    Links a tokenizer to a dataset hash and captures training intent.
    """

    name: str
    vocab_size: int
    model_type: str = "bpe"           # e.g., bpe, unigram, wordpiece
    type: str | None = None           # alias for model_type
    dataset_hash: Optional[str] = None

    # Normalization / pretokenization hints
    lowercase: bool = True
    byte_level: bool = True
    strip_accents: bool = False
    add_prefix_space: bool = False

    # Special tokens
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"

    # Versioning
    version: str = "v1"

    def validate(self) -> List[str]:
        errors: List[str] = []

        if self.vocab_size <= 0:
            errors.append("vocab_size must be positive")

        if not self.name:
            errors.append("name is required")

        if self.dataset_hash is None:
            errors.append("dataset_hash must be provided to bind tokenizer to dataset")

        return errors
