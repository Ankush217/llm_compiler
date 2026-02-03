from pathlib import Path
import time
# ---- IMPORT YOUR SYSTEM ----
from llm_compiler import (
    DatasetSpec,
    DatasetCompiler,

    TokenizerSpec,
    TokenizerCompiler,

    LLM,
    compile_spec,

    OptimizerSpec,
    TrainingSpec,

    run_training,
)

# ---------------------------------------------------------------------
# 1. DATASET (tiny, synthetic)
# ---------------------------------------------------------------------
dataset_spec = DatasetSpec(
    name="debug_dataset",
    target_tokens=1_000,                # tiny
    languages={"en": 1.0},
    domains=["synthetic"],
    min_length=4,
    max_length=16,
)

dataset_ir = DatasetCompiler().compile(dataset_spec)
dataset_hash = dataset_ir.fingerprint()

# ---------------------------------------------------------------------
# 2. TOKENIZER (tiny vocab)
# ---------------------------------------------------------------------
tokenizer_spec = TokenizerSpec(
    name="debug_tokenizer",
    type="bpe",
    vocab_size=32,                      # very small
    dataset_hash=dataset_hash,
)

tokenizer_ir = TokenizerCompiler().compile(tokenizer_spec)
tokenizer_hash = tokenizer_ir.fingerprint()

# ---------------------------------------------------------------------
# 3. MODEL (~2K PARAM TRANSFORMER)
# ---------------------------------------------------------------------
# This config intentionally produces ~1–3k params
llm_spec = LLM(
    name="debug-2k-transformer",
    template="decoder_only",

    vocab_size=32,
    context_length=16,

    num_layers=1,
    hidden_size=16,
    num_heads=2,
    intermediate_size=32,

    activation="relu",
    norm="rmsnorm",

    tie_embeddings=True,
)

model_ir = compile_spec(llm_spec)
model_hash = model_ir.fingerprint()

print("Model params:", model_ir.parameter_count)

# ---------------------------------------------------------------------
# 4. OPTIMIZER (simple)
# ---------------------------------------------------------------------
optimizer_spec = OptimizerSpec(
    type="adamw",
    lr=1e-3,
    weight_decay=0.0,
)

# ---------------------------------------------------------------------
# 5. TRAINING SPEC (VERY SMALL)
# ---------------------------------------------------------------------
training_spec = TrainingSpec(
    name="debug_run",

    model_hash=model_hash,
    dataset_hash=dataset_hash,
    tokenizer_hash=tokenizer_hash,

    optimizer=optimizer_spec,

    batch_size=2,
    microbatch_size=1,
    max_steps=10,                       # very short run
)

training_ir = training_spec.compile()
training_hash = training_ir.fingerprint()

# ---------------------------------------------------------------------
# 6. EXECUTE (REAL PIPELINE)
# ---------------------------------------------------------------------
t0 = time.perf_counter()

result = run_training(
    training_ir=training_ir,
    model_ir=model_ir,
    dataset_ir=dataset_ir,
    tokenizer_ir=tokenizer_ir,
    output_root=Path("runs"),
)

t1 = time.perf_counter()

print(f"\nTotal execution time: {(t1 - t0)*1000:.2f} ms")

# ---------------------------------------------------------------------
# 7. DEBUG OUTPUT
# ---------------------------------------------------------------------
print("\n=== DEBUG RUN COMPLETE ===")
print("Training hash:", result.training_hash)
print("Status:", result.status)
print("Artifacts:")
for k, v in result.artifacts.items():
    print(" ", k, "→", v)
print("Metrics:", result.metrics)
print("Logs:")
for line in result.logs:
    print(" ", line)
