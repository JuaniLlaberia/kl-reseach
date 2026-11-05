import copy
import json
from pathlib import Path

import torch
import transformers
from transformers import GenerationConfig

# Every value must be explicit: generate() inherits from the model's generation config
# any field left out or set to None (Qwen ships top_k=20, repetition_penalty=1.1, etc.).
SAMPLING_SETTINGS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
}

def build_prompt_ids(tokenizer, prompts: list[dict]) -> list[dict]:
    """
    Applies the chat template to the prompts, using each family's default system prompt.

    Args:
        tokenizer: Tokenizer of the base model.
        prompts (list[dict]): Prompts with the fields prompt_id and text.
    Returns:
        List of prompts with the fields prompt_id and prompt_ids.
    """
    return [
        {
            "prompt_id": prompt["prompt_id"],
            "prompt_ids": list(tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["text"]}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )),
        }
        for prompt in prompts
    ]

def get_stop_ids(model) -> list[int]:
    """
    Gets the token IDs that end a response for this model.

    Args:
        model: Model used for sampling.
    Returns:
        List of stop token IDs.
    """
    stop_ids = model.generation_config.eos_token_id
    return [stop_ids] if isinstance(stop_ids, int) else list(stop_ids)

def build_generation_config(max_new_tokens: int, stop_ids: list[int], pad_id: int) -> GenerationConfig:
    """
    Builds the generation config used to sample responses from the model's distribution.

    Args:
        max_new_tokens (int): Maximum number of tokens per response.
        stop_ids (list[int]): Token IDs that end a response.
        pad_id (int): Token ID used for padding.
    Returns:
        A generation config.
    """
    return GenerationConfig(**SAMPLING_SETTINGS, max_new_tokens=max_new_tokens, eos_token_id=stop_ids, pad_token_id=pad_id)

def check_unmodified_sampling(model, prompt_ids: list[int], generation_config: GenerationConfig) -> None:
    """
    Fails if generate() changes the model's logits before sampling from them.

    Args:
        model: Model used for sampling.
        prompt_ids (list[int]): Token IDs of any prompt.
        generation_config (GenerationConfig): Generation config to check.
    Returns:
        None.
    """
    config = copy.deepcopy(generation_config)
    config.max_new_tokens = 8
    config.output_scores = True
    config.output_logits = True
    config.return_dict_in_generate = True

    input_ids = torch.tensor([prompt_ids], device=model.device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=config)

    # scores are the processed logits, logits are the raw ones
    for processed, raw in zip(output.scores, output.logits):
        if not torch.equal(processed, raw):
            raise ValueError("generate() is changing the logits, so samples wouldn't come from the model's distribution.")

def left_pad(batch_ids: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Left-pads a batch of prompts for generation.

    Args:
        batch_ids (list[list[int]]): Token IDs of each prompt.
        pad_id (int): Token ID used for padding.
    Returns:
        The padded token IDs and their attention mask.
    """
    max_length = max(len(ids) for ids in batch_ids)
    input_ids = torch.tensor([[pad_id] * (max_length - len(ids)) + ids for ids in batch_ids])
    attention_mask = torch.tensor([[0] * (max_length - len(ids)) + [1] * len(ids) for ids in batch_ids])
    return input_ids, attention_mask

def extract_response(generated_ids: list[int], stop_ids: list[int]) -> tuple[list[int], bool]:
    """
    Cuts a generated sequence right after its first stop token, which counts as a response token.

    Args:
        generated_ids (list[int]): Token IDs generated after the prompt, including any padding.
        stop_ids (list[int]): Token IDs that end a response.
    Returns:
        The response token IDs, and whether the response hit the length cap without stopping.
    """
    for index, token_id in enumerate(generated_ids):
        if token_id in stop_ids:
            return generated_ids[:index + 1], False
    return generated_ids, True

def load_responses(path: Path) -> tuple[dict, list[dict]]:
    """
    Loads a saved responses file.

    Args:
        path (Path): Path to the JSONL file (the first line is the header).
    Returns:
        The header and the list of response records.
    """
    with open(path, encoding="utf-8") as file:
        lines = [json.loads(line) for line in file]
    return lines[0]["header"], lines[1:]

def sample_responses(model, tokenizer, prompts: list[dict], out_path: Path, seed: int, batch_size: int, max_new_tokens: int, model_info: dict) -> None:
    """
    Samples one response per prompt and saves them batch by batch, resuming from out_path if it already exists.

    Args:
        model: Model to sample from, already loaded in its final precision and device.
        tokenizer: Tokenizer of the model.
        prompts (list[dict]): Prompts with the fields prompt_id and prompt_ids, in a fixed order.
        out_path (Path): Path of the JSONL file to write.
        seed (int): Base seed. Each batch uses seed + batch index.
        batch_size (int): Number of prompts per batch.
        max_new_tokens (int): Maximum number of tokens per response.
        model_info (dict): Model name and revision or checkpoint, saved in the header.
    Returns:
        None.
    """
    model.eval()
    stop_ids = get_stop_ids(model)
    pad_id = tokenizer.pad_token_id
    generation_config = build_generation_config(max_new_tokens, stop_ids, pad_id)
    check_unmodified_sampling(model, prompts[0]["prompt_ids"], generation_config)

    header = {
        "model_info": model_info,
        "seed": seed,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "stop_ids": stop_ids,
        "dtype": str(model.dtype),
        "settings": SAMPLING_SETTINGS,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }

    completed = 0
    if out_path.exists():
        saved_header, records = load_responses(out_path)
        if saved_header != header:
            raise ValueError(f"{out_path} was written with a different setup, so it can't be resumed.")
        completed = len(records)
        if completed % batch_size != 0 and completed != len(prompts):
            raise ValueError(f"{out_path} ends in the middle of a batch, so it can't be resumed.")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(json.dumps({"header": header}) + "\n")

    for start in range(completed, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        input_ids, attention_mask = left_pad([prompt["prompt_ids"] for prompt in batch], pad_id)

        # The seed depends on the batch position, so a resumed run gets the same samples
        torch.manual_seed(seed + start // batch_size)
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                generation_config=generation_config,
            )

        generated = sequences[:, input_ids.shape[1]:].tolist()
        with open(out_path, "a", encoding="utf-8") as file:
            for prompt, generated_ids in zip(batch, generated):
                response_ids, truncated = extract_response(generated_ids, stop_ids)
                file.write(json.dumps({
                    "prompt_id": prompt["prompt_id"],
                    "prompt_ids": prompt["prompt_ids"],
                    "response_ids": response_ids,
                    "text": tokenizer.decode(response_ids, skip_special_tokens=True),
                    "truncated": truncated,
                }) + "\n")
