from .spec import TokenizerSpec
from .ir import TokenizerIR


class TokenizerCompiler:
    """
    Compiles TokenizerSpec -> TokenizerIR (intent-only).
    """

    def compile(self, spec: TokenizerSpec) -> TokenizerIR:
        errors = spec.validate()
        if errors:
            raise ValueError(f"TokenizerSpec invalid: {errors}")

        ir = TokenizerIR(name=spec.name, version=spec.version)
        model_type = spec.model_type or spec.type or "bpe"
        ir.metadata["vocab_size"] = spec.vocab_size
        ir.metadata["model_type"] = model_type
        ir.metadata["dataset_hash"] = spec.dataset_hash
        ir.metadata["lowercase"] = spec.lowercase
        ir.metadata["byte_level"] = spec.byte_level
        ir.metadata["strip_accents"] = spec.strip_accents
        ir.metadata["add_prefix_space"] = spec.add_prefix_space
        ir.metadata["special_tokens"] = {
            "bos": spec.bos_token,
            "eos": spec.eos_token,
            "unk": spec.unk_token,
            "pad": spec.pad_token,
        }

        # Nodes capturing high-level phases
        ir.add_node("normalize", lowercase=spec.lowercase, strip_accents=spec.strip_accents)
        ir.add_node("tokenize", model_type=model_type, vocab_size=spec.vocab_size)
        ir.add_node("special_tokens", tokens=ir.metadata["special_tokens"])

        return ir
