# llm_compiler

A **compiler-style system for building, validating, and training Large Language Models (LLMs)** from a single declarative specification.

This project treats LLMs as **compiled artifacts**, not hand-written model code.

---

## 🚀 What is this?

`llm_compiler` is an **end-to-end LLM compiler pipeline** that takes a high-level spec and produces:

- ✅ A validated model architecture
- ✅ A framework-agnostic Intermediate Representation (IR)
- ✅ Tokenizer + dataset pipelines
- ✅ Parameter & constraint solving
- ✅ Training plans and execution runs
- ✅ Reproducible artifacts (hashes, manifests, provenance)

All from **one Python file**.

---

## 🧠 Core Philosophy

- **Architecture-first**, not checkpoint-first
- **Compiler mindset**, not ad-hoc model code
- **Explicit over implicit** (no magic defaults)
- **Reproducibility by construction**
- **Separation of concerns**:
  - Spec → IR → Emit → Train → Run

---

## 🧩 Major Components

- **Specification Layer**
  - `LLM`, `DatasetSpec`, `TokenizerSpec`, `TrainingSpec`
- **Constraint & Dimension Solver**
  - Solves hidden size, heads, layers, etc.
- **Intermediate Representation (IR)**
  - Graph-based, framework-agnostic
- **Emitters**
  - PyTorch, Safetensors, TorchScript (extensible)
- **Execution Engine**
  - Tokenization, training, checkpoints, metrics
- **Provenance & Hashing**
  - Every run is reproducible and traceable

---

## 🧪 Example (Tiny End-to-End Run)

```python
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

dataset_spec = DatasetSpec(
    name="debug_dataset",
    target_tokens=1_000,                
    languages={"en": 1.0},
    domains=["synthetic"],
    min_length=4,
    max_length=16,
)

dataset_ir = DatasetCompiler().compile(dataset_spec)
dataset_hash = dataset_ir.fingerprint()

tokenizer_spec = TokenizerSpec(
    name="debug_tokenizer",
    type="bpe",
    vocab_size=32,                      
    dataset_hash=dataset_hash,
)

tokenizer_ir = TokenizerCompiler().compile(tokenizer_spec)
tokenizer_hash = tokenizer_ir.fingerprint()

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


optimizer_spec = OptimizerSpec(
    type="adamw",
    lr=1e-3,
    weight_decay=0.0,
)

training_spec = TrainingSpec(
    name="debug_run",

    model_hash=model_hash,
    dataset_hash=dataset_hash,
    tokenizer_hash=tokenizer_hash,

    optimizer=optimizer_spec,

    batch_size=2,
    microbatch_size=1,
    max_steps=10,
)

training_ir = training_spec.compile()
training_hash = training_ir.fingerprint()
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
```

This builds a fully valid transformer with ~2K parameters.

## 📦 Current Status

✔ Decoder-only transformers

✔ Dataset + tokenizer compilation

✔ Training execution (real & debug)

✔ Reproducible run directories

✔ Parameter-accurate IR validation

This project is actively evolving.

## 🧑‍💻 Why this exists

Most LLM tooling focuses on using models.

This project focuses on defining, compiling, and reasoning about models themselves.
