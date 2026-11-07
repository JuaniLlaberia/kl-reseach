import gc
import json
import subprocess
import time
from pathlib import Path

import lm_eval
import numpy as np
import torch
import transformers
from lm_eval.models.huggingface import HFLM

# Moved from notebooks/stage-1.ipynb without changes. MODEL_NAME and MODEL_REVISION stay for that record;
# run_retention takes the model as an argument, so they don't decide what gets evaluated.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EVAL_RUNS = [
    {"task": "hellaswag", "num_fewshot": 0, "batch_size": 8},
    {"task": "truthfulqa_mc2", "num_fewshot": 0, "batch_size": 8},
    {"task": "winogrande", "num_fewshot": 5, "batch_size": 8},
    {"task": "mmlu", "num_fewshot": 5, "batch_size": 8},
    {"task": "ifeval", "num_fewshot": 0, "batch_size": 8},
    {"task": "humaneval_instruct", "num_fewshot": 0, "batch_size": 8},
]
DTYPE = "float32"
CHAT_TEMPLATE = True
DEVICE = "cuda:0"
GEN_KWARGS = {"repetition_penalty": 1.0}

def bootstrap_ci(values: list[float], iters: int = 10000, seed: int = 0) -> list[float]:
    """
    Computes a percentile bootstrap CI of the mean. lm-eval only reports a closed-form standard error for it.

    Args:
        values (list[float]): Score of each sample.
        iters (int): Number of resamples.
        seed (int): Seed of the resampling.
    Returns:
        The lower and upper bounds of the 95% CI.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(iters)]
    return [float(bound) for bound in np.percentile(means, [2.5, 97.5])]

def summarize_samples(samples: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
    """
    Computes the score and CI of each metric, pooling subtasks (MMLU's subjects) so each question weighs the same.

    Args:
        samples (dict[str, list[dict]]): Per-sample output of lm-eval, by subtask.
    Returns:
        The score and CI of each metric, and one row per sample without its prompt.
    """
    rows = []
    for subtask, task_samples in samples.items():
        for sample in task_samples:
            row = {"subtask": subtask, "doc_id": sample["doc_id"]}
            # Aggregate-only metrics, like IFEval's instruction-level scores, aren't one number per sample
            row.update({m: sample[m] for m in sample["metrics"] if isinstance(sample[m], (bool, int, float))})
            response = sample["resps"][0][0]
            if isinstance(response, str):
                row["response"] = response
            rows.append(row)

    metrics = [name for name in rows[0] if name not in ("subtask", "doc_id", "response")]
    scores = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        scores[metric] = {"mean": float(np.mean(values)), "ci95": bootstrap_ci(values), "n": len(values)}

    return scores, rows

def write_json(path: Path, content: dict) -> None:
    """
    Writes a JSON file in one step, so a crash can't leave it half written.

    Args:
        path (Path): Path of the file.
        content (dict): Content of the file.
    Returns:
        None.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)

def run_retention(model_path: str, out_dir: str, run: str, checkpoint: str, revision: str | None = None, limit: int | None = None) -> dict:
    """
    Runs the retention suite on one checkpoint. Saves after each benchmark and skips the ones already saved.

    Args:
        model_path (str): Hugging Face model name or local checkpoint path.
        out_dir (str): Folder of this run's results.
        run (str): Run name.
        checkpoint (str): Checkpoint name, which names the results file.
        revision (str | None): Hugging Face commit of the model, if it has one.
        limit (int | None): Documents per task, for smoke tests only. Never for reported scores.
    Returns:
        The content of the results file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{checkpoint}.json"

    setup = {
        "run": run,
        "checkpoint": checkpoint,
        "model": model_path,
        "revision": revision,
        "dtype": DTYPE,
        "chat_template": CHAT_TEMPLATE,
        "gen_kwargs": GEN_KWARGS,
        "eval_runs": EVAL_RUNS,
        "limit": limit,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_version": lm_eval.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(DEVICE),
    }

    out = {**setup, "retention": {}}
    if out_path.exists():
        out = json.loads(out_path.read_text(encoding="utf-8"))
        different = [key for key, value in setup.items() if out.get(key) != value]
        if different:
            raise ValueError(f"{out_path} was written with a different setup ({different}), so it can't be resumed.")

    lm = HFLM(pretrained=model_path, revision=revision, dtype=DTYPE, batch_size=EVAL_RUNS[0]["batch_size"], device=DEVICE)

    for eval_run in EVAL_RUNS:
        task = eval_run["task"]
        if task in out["retention"]:
            print(f"{task}: already saved, skipping")
            continue

        lm.batch_size_per_gpu = eval_run["batch_size"]
        start = time.perf_counter()
        res = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=eval_run["num_fewshot"],
            apply_chat_template=CHAT_TEMPLATE,
            fewshot_as_multiturn=CHAT_TEMPLATE and eval_run["num_fewshot"] > 0,
            gen_kwargs=GEN_KWARGS,
            random_seed=0,
            numpy_random_seed=0,
            torch_random_seed=0,
            fewshot_random_seed=0,
            confirm_run_unsafe_code=True,
            log_samples=True,
            limit=limit,
        )
        minutes = (time.perf_counter() - start) / 60

        scores, rows = summarize_samples(res["samples"])
        with open(out_dir / f"{checkpoint}_{task}_samples.jsonl", "w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")

        out["retention"][task] = {
            "minutes": minutes,
            "scores": scores,
            "harness_results": res["results"],
            "harness_groups": res.get("groups", {}),
        }
        write_json(out_path, out)
        print(f"{task}: {minutes:.1f} min | saved")

    del lm
    gc.collect()
    torch.cuda.empty_cache()
    return out
