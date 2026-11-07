import copy
import string

import pytest
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

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

CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

def build_chat_tokenizer() -> PreTrainedTokenizerFast:
    """Character-level tokenizer with a Qwen-style chat template, so decoding gives back the real text with no download."""
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = {token: index for index, token in enumerate(specials + list(string.printable))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
    tokenizer.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast

def build_chat_model(tokenizer) -> Qwen2ForCausalLM:
    """Tiny Qwen2 model in fp32, which ships Qwen's own sampling settings and stop tokens."""
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        use_sliding_window=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).float()
    model.generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.pad_token_id,
    )
    return model

def make_row(row_id: int, gold: str = "B", kind: str = "mcq-4-choices", task: str = "task_a") -> dict:
    """A dataset row in the format of src/data/chem.py."""
    row = {
        "row_id": row_id,
        "question_hash": f"hash{row_id}",
        "type": kind,
        "question": f"Question {row_id}?",
        "details": {"task": task, "subtask": task},
    }
    if kind == "mcq-4-choices":
        return {**row, "choices": {"label": ["A", "B", "C", "D"], "text": ["one", "two", "three", "four"]}, "answerKey": gold, "answer": ""}
    return {**row, "answer": gold, "answerKey": ""}

@pytest.fixture
def chat_tokenizer():
    return build_chat_tokenizer()

@pytest.fixture
def chat_model(chat_tokenizer):
    return build_chat_model(chat_tokenizer)

@pytest.fixture
def solved_rows():
    """Sixteen solved four-choice rows."""
    return [{**make_row(index), "solution": "The second one fits.\nAnswer: B", "attempt": 1} for index in range(16)]

@pytest.fixture
def model_dir(chat_tokenizer, chat_model, tmp_path):
    """A saved tiny model, so run_config can load it by path as it loads a real one."""
    path = tmp_path / "tiny_model"
    chat_model.save_pretrained(path)
    chat_tokenizer.save_pretrained(path)
    return path
