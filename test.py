from pathlib import Path

from llm_compiler.spec import LLM
from llm_compiler.compile import compile_spec

def main():
    spec = LLM(
        template="decoder_only_v1",

        # choose ONE of these
        target_params=125_000_000,
        # explicit_dims={"num_layers": 12, "hidden_size": 768},

        vocab_size=32000,
        context_length=2048,

        attention="gqa",
        norm="rmsnorm",
        activation="swiglu",
        positional_encoding="rope",

        tokenizer="unigram",
        weight_format="safetensors",
        backend="pytorch_training",

        precision="float32",
        tie_word_embeddings=True,
    )

    out_dir = Path("out_model")

    report = compile_spec(
        spec=spec,
        output_dir=out_dir,
        verbose=True,   # 👈 THIS is why you saw nothing before
    )

    print("\n=== COMPILATION REPORT ===")
    print("Success:", report["success"])
    print("Model name:", report.get("model_name"))
    print("Parameters:", report.get("solution", {}).get("actual_params"))
    print("Output dir:", report["output_dir"])

if __name__ == "__main__":
    main()
