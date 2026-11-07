from typing import Callable

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from src.data.chem import build_prompt, get_gold
from src.eval.extract import is_correct
from src.kl.sampling import get_stop_ids
from src.train.common import check_fp32, training_arguments

# Dr. GRPO, with binary rewards that are only centered per group. TRL's defaults are DAPO with rewards scaled per group.
LOSS_TYPE = "dr_grpo"
SCALE_REWARDS = False
BETA = 0.0
# The same distribution that KL and the new-task eval sample from, set in src/kl/sampling.py
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
REPETITION_PENALTY = 1.0

class ChunkedGRPOTrainer(GRPOTrainer):
    """GRPO that generates its completions in chunks. TRL generates the whole batch at once, which can run out of memory."""

    def __init__(self, *args, generation_chunk_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_chunk_size = generation_chunk_size

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        completion_ids = []
        for start in range(0, len(prompt_ids), self.generation_chunk_size):
            chunk = prompt_ids[start:start + self.generation_chunk_size]
            ids, _, _ = super()._generate_single_turn(chunk, images, multimodal_fields, num_generations, has_tool_images)
            completion_ids.extend(ids)
        return completion_ids, None, None

def build_reward(stop_ids: list[int]) -> Callable:
    """
    Builds the reward: 1 if a completion is correct and 0 otherwise, with no format reward.
    It's the same is_correct as the eval and the GPT-4o solutions, so a response that hit the length cap gets 0.

    Args:
        stop_ids (list[int]): Token IDs that end a response.
    Returns:
        The reward function, which receives the gold and answer_type columns from the dataset.
    """
    def correctness_reward(prompts, completions, completion_ids, gold, answer_type, **kwargs) -> list[float]:
        return [
            float(is_correct(completion[0]["content"], gold_answer, kind, truncated=not ids or ids[-1] not in stop_ids))
            for completion, ids, gold_answer, kind in zip(completions, completion_ids, gold, answer_type)
        ]
    return correctness_reward

def build_grpo_dataset(solved_rows: list[dict]) -> Dataset:
    """
    Builds the training data from the same rows as SFT. The trainer applies the chat template to each prompt.

    Args:
        solved_rows (list[dict]): Rows with a solution. Only their prompts and gold answers are used.
    Returns:
        A dataset with the prompt, gold and answer_type columns.
    """
    return Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": build_prompt(row)}],
            "gold": get_gold(row),
            "answer_type": row["type"],
        }
        for row in solved_rows
    ])

def build_grpo_args(config: dict, out_dir, report_to: str = "none") -> GRPOConfig:
    """
    Builds GRPO's settings. Each generation is used for one optimizer step, split over micro-batches to fit in memory,
    so the weights don't change between sampling and the loss. That keeps the importance ratios at 1 and clipping inactive.

    Args:
        config (dict): Settings of the run.
        out_dir: Folder for the trainer's own files.
        report_to (str): "wandb" or "none".
    Returns:
        The trainer's config.
    """
    micro_batches = config["group_size"] * config["prompts_per_step"] // config["loss_batch_size"]
    return GRPOConfig(
        **training_arguments(config, out_dir, report_to),
        per_device_train_batch_size=config["loss_batch_size"],
        gradient_accumulation_steps=micro_batches,
        steps_per_generation=micro_batches,
        num_iterations=1,
        num_generations=config["group_size"],
        max_completion_length=config["max_completion_length"],
        loss_type=LOSS_TYPE,
        scale_rewards=SCALE_REWARDS,
        beta=BETA,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        mask_truncated_completions=False,
    )

def build_grpo_trainer(config: dict, model, tokenizer, solved_rows: list[dict], out_dir, report_to: str = "none") -> ChunkedGRPOTrainer:
    """
    Builds GRPO, fine-tuning every weight with no adapter. The model is passed already loaded, so its precision is explicit.

    Args:
        config (dict): Settings of the run.
        model: Model in fp32, with every weight trainable.
        tokenizer: Tokenizer of the model.
        solved_rows (list[dict]): Rows with a solution, so GRPO trains on the same prompts as SFT.
        out_dir: Folder for the trainer's own files.
        report_to (str): "wandb" or "none".
    Returns:
        The trainer.
    """
    check_fp32(model)
    stop_ids = get_stop_ids(model)
    trainer = ChunkedGRPOTrainer(
        model=model,
        reward_funcs=[build_reward(stop_ids)],
        args=build_grpo_args(config, out_dir, report_to),
        train_dataset=build_grpo_dataset(solved_rows),
        processing_class=tokenizer,
        generation_chunk_size=config["generation_chunk_size"],
    )
    # TRL also stops at the tokenizer's end token, which would make its responses differ from the ones KL measures
    if set(trainer.eos_token_ids) != set(stop_ids):
        raise ValueError(f"GRPO stops at {trainer.eos_token_ids}, but KL's responses stop at {stop_ids}.")
    return trainer
