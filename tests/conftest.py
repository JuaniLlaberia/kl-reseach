import copy

import pytest
import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from src.kl.sampling import sample_responses

STOP_IDS = [199, 198]
MAX_NEW_TOKENS = 30
BATCH_SIZE = 32

class FakeTokenizer:
    """Tokenizer with the only two things the sampler needs, so tests don't need a download."""
    pad_token_id = 0

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(token_id) for token_id in ids)

@pytest.fixture(scope="session")
def tokenizer():
    return FakeTokenizer()

@pytest.fixture(scope="session")
def base_model():
    """Tiny random Qwen2 model, shipping the same sampling settings as the real Qwen2.5-0.5B-Instruct."""
    config = Qwen2Config(
        vocab_size=200,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        use_sliding_window=False,
        pad_token_id=0,
        eos_token_id=STOP_IDS[0],
        bos_token_id=0,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).eval()
    model.generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,
        eos_token_id=STOP_IDS,
        pad_token_id=0,
        bos_token_id=0,
    )
    return model

@pytest.fixture(scope="session")
def ft_model(base_model):
    """The base model with noise added to its weights, standing in for a fine-tuned checkpoint."""
    model = copy.deepcopy(base_model)
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    return model

@pytest.fixture(scope="session")
def prompts():
    """300 prompts with varied lengths, so batches need padding."""
    generator = torch.Generator().manual_seed(5)
    lengths = torch.randint(2, 9, (300,), generator=generator).tolist()
    return [
        {"prompt_id": index, "prompt_ids": torch.randint(1, 190, (length,), generator=generator).tolist()}
        for index, length in enumerate(lengths)
    ]

@pytest.fixture(scope="session")
def base_path(base_model, tokenizer, prompts, tmp_path_factory):
    path = tmp_path_factory.mktemp("responses") / "base.jsonl"
    sample_responses(base_model, tokenizer, prompts, path, 7, BATCH_SIZE, MAX_NEW_TOKENS, {"model": "base"})
    return path

@pytest.fixture(scope="session")
def ft_path(ft_model, tokenizer, prompts, tmp_path_factory):
    path = tmp_path_factory.mktemp("responses") / "ft.jsonl"
    sample_responses(ft_model, tokenizer, prompts, path, 11, BATCH_SIZE, MAX_NEW_TOKENS, {"model": "ft"})
    return path
