from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetSpec:
    """
    Declarative dataset specification.
    Defines intent only (no IO, no crawling).
    """

    name: str

    # Size intent
    target_tokens: int

    # High-level composition
    languages: Dict[str, float]  # {"en": 0.9, "hi": 0.1}
    domains: List[str]           # ["wikipedia", "books", "code"]

    # Structural filters
    min_length: int = 100
    max_length: int = 8192

    # Quality flags
    deduplicate: bool = True
    remove_boilerplate: bool = True

    # Versioning
    version: str = "v1"

    def validate(self) -> List[str]:
        errors: List[str] = []

        if self.target_tokens <= 0:
            errors.append("target_tokens must be positive")

        if not self.languages:
            errors.append("languages cannot be empty")
        else:
            total = sum(self.languages.values())
            if abs(total - 1.0) > 1e-6:
                errors.append("language ratios must sum to 1.0")

        if not self.domains:
            errors.append("domains cannot be empty")

        if self.min_length >= self.max_length:
            errors.append("min_length must be < max_length")

        return errors
