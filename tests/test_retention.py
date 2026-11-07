import ast
import json
from pathlib import Path

import pytest

from src.eval import retention

MOVED_FUNCTIONS = ("bootstrap_ci", "summarize_samples", "write_json", "run_retention")
MOVED_CONSTANTS = ("EVAL_RUNS", "DTYPE", "CHAT_TEMPLATE", "DEVICE", "GEN_KWARGS")

def stage_1_cells() -> str:
    notebook = json.loads(Path("notebooks/stage-1.ipynb").read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"][8:10] if cell["cell_type"] == "code")

def definitions(source: str) -> dict[str, str]:
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign):
            found[node.targets[0].id] = ast.dump(node.value)
    return found

@pytest.mark.parametrize("name", MOVED_FUNCTIONS + MOVED_CONSTANTS)
def test_moved_without_changes(name):
    module = definitions(Path("src/eval/retention.py").read_text(encoding="utf-8"))
    assert module[name] == definitions(stage_1_cells())[name]

def test_summary_pools_subtasks_and_leaves_out_prompts():
    samples = {
        "a": [{"doc_id": 0, "metrics": ["acc"], "acc": 1.0, "resps": [["x"]]}, {"doc_id": 1, "metrics": ["acc"], "acc": 0.0, "resps": [["y"]]}],
        "b": [{"doc_id": 0, "metrics": ["acc"], "acc": 1.0, "resps": [[("loglik", True)]]}],
    }
    scores, rows = retention.summarize_samples(samples)
    assert scores["acc"]["mean"] == pytest.approx(2 / 3)
    assert scores["acc"]["n"] == 3
    assert rows[0] == {"subtask": "a", "doc_id": 0, "acc": 1.0, "response": "x"}
    assert "response" not in rows[2]

def test_bootstrap_is_reproducible_and_brackets_the_mean():
    low, high = retention.bootstrap_ci([0.0, 1.0] * 50)
    assert low < 0.5 < high
    assert [low, high] == retention.bootstrap_ci([0.0, 1.0] * 50)

def test_write_json_leaves_no_temporary_file(tmp_path):
    retention.write_json(tmp_path / "out.json", {"a": 1})
    assert json.loads((tmp_path / "out.json").read_text()) == {"a": 1}
    assert [path.name for path in tmp_path.iterdir()] == ["out.json"]
