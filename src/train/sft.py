import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from src.data.chem import build_prompt
from src.kl.sampling import build_prompt_ids, get_stop_ids
from src.train.common import check_fp32, training_arguments

IGNORE_INDEX = -100

def build_sft_rows(tokenizer, solved_rows: list[dict], stop_id: int, max_length: int) -> list[dict]:
    """
    Tokenizes the prompts and solutions. Prompts use the same IDs as the KL measurement, and the labels end at the
    first stop token, as a sampled response does. The chat template's newline after the stop token is left out.

    Args:
        tokenizer: Tokenizer of the model.
        solved_rows (list[dict]): Rows with a solution.
        stop_id (int): Token that ends a response.
        max_length (int): Longest allowed sequence. Longer rows raise an error, as cutting them would remove their answer.
    Returns:
        One example per row, with input_ids and labels. Prompt tokens are masked in the labels.
    """
    prompts = build_prompt_ids(tokenizer, [{"prompt_id": row["row_id"], "text": build_prompt(row)} for row in solved_rows])
    examples = []
    too_long = []
    for row, prompt in zip(solved_rows, prompts):
        response_ids = tokenizer(row["solution"], add_special_tokens=False)["input_ids"] + [stop_id]
        input_ids = prompt["prompt_ids"] + response_ids
        if len(input_ids) > max_length:
            too_long.append(row["row_id"])
        examples.append({"input_ids": input_ids, "labels": [IGNORE_INDEX] * len(prompt["prompt_ids"]) + response_ids})
    if too_long:
        raise ValueError(f"{len(too_long)} rows are longer than max_length={max_length}, for example {too_long[:5]}.")
    return examples

def collate_sft(features: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """
    Right-pads a batch. Padded positions are masked in both the attention and the labels.

    Args:
        features (list[dict]): Examples with input_ids and labels.
        pad_id (int): Token ID used for padding.
    Returns:
        The input_ids, attention_mask and labels of the batch.
    """
    longest = max(len(feature["input_ids"]) for feature in features)
    pad = lambda values, value: values + [value] * (longest - len(values))
    return {
        "input_ids": torch.tensor([pad(feature["input_ids"], pad_id) for feature in features]),
        "attention_mask": torch.tensor([pad([1] * len(feature["input_ids"]), 0) for feature in features]),
        "labels": torch.tensor([pad(feature["labels"], IGNORE_INDEX) for feature in features]),
    }

def build_sft_trainer(config: dict, model, tokenizer, solved_rows: list[dict], out_dir, report_to: str = "none") -> Trainer:
    """
    Builds SFT with a plain Trainer. The loss is the model's own cross-entropy, averaged over the response tokens of each
    micro-batch. TRL's SFTTrainer would add templating and truncation that this data doesn't need.

    Args:
        config (dict): Settings of the run.
        model: Model in fp32, with every weight trainable.
        tokenizer: Tokenizer of the model.
        solved_rows (list[dict]): Rows with a solution.
        out_dir: Folder for the trainer's own files.
        report_to (str): "wandb" or "none".
    Returns:
        The trainer.
    """
    check_fp32(model)
    stop_ids = get_stop_ids(model)
    if tokenizer.eos_token_id not in stop_ids:
        raise ValueError(f"The tokenizer's end token {tokenizer.eos_token_id} isn't one of the model's stop tokens {stop_ids}.")

    examples = build_sft_rows(tokenizer, solved_rows, tokenizer.eos_token_id, config["max_length"])
    args = TrainingArguments(
        **training_arguments(config, out_dir, report_to),
        per_device_train_batch_size=config["micro_batch_size"],
        gradient_accumulation_steps=config["batch_size"] // config["micro_batch_size"],
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(examples),
        data_collator=lambda features: collate_sft(features, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
