
import torch
from .config import ModelConfig
from .llmc-decoder-only-0m-relu import llmc-decoder-only-0m-relu


def test_model():
    config = ModelConfig()
    model = llmc-decoder-only-0m-relu(config)
    batch_size = 2
    seq_len = min(16, config.max_position_embeddings)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    outputs = model(input_ids=input_ids)
    if isinstance(outputs, dict):
        logits = list(outputs.values())[0]
    else:
        logits = outputs
    assert logits.shape[:2] == (batch_size, seq_len)
    print("Model forward succeeded; output shape", logits.shape)


if __name__ == "__main__":
    test_model()
