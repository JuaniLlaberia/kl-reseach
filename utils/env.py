import importlib.metadata as metadata
import re
import subprocess
from pathlib import Path

def read_pins(requirements_path: Path) -> dict[str, str]:
    """
    Reads the pinned versions, without any extras (lm-eval[hf,ifeval]==0.4.13 becomes lm-eval).

    Args:
        requirements_path (Path): Path of requirements.txt.
    Returns:
        Dictionary from package name to version.
    """
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return dict(re.sub(r"\[.*\]", "", line.strip()).split("==") for line in lines if "==" in line)

def check_pins(requirements_path: Path) -> dict[str, str]:
    """
    Fails if any installed package isn't on its pinned version.

    Args:
        requirements_path (Path): Path of requirements.txt.
    Returns:
        Dictionary from package name to installed version.
    """
    pins = read_pins(requirements_path)
    installed = {name: metadata.version(name).split("+")[0] for name in pins}
    wrong = {name: (installed[name], pins[name]) for name in pins if installed[name] != pins[name]}
    if wrong:
        raise RuntimeError(f"Not on the pinned versions (installed, pinned): {wrong}")
    return installed

def git(repo: Path, *args: str) -> str:
    """
    Runs a git command in a repo.

    Args:
        repo (Path): Path of the repo.
        args (str): Arguments of the command.
    Returns:
        The command's output.
    """
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

def check_pushed(repo: Path) -> str:
    """
    Fails unless the code is exactly what's on the remote, so every result traces to a pushed commit.
    Results and notebook outputs live outside the repo during a run, so anything changed here is a real difference.

    Args:
        repo (Path): Path of the repo.
    Returns:
        The current commit.
    """
    changed = git(repo, "status", "--porcelain")
    if changed:
        raise RuntimeError(f"The repo has uncommitted changes:\n{changed}")

    git(repo, "fetch", "--quiet", "origin")
    head = git(repo, "rev-parse", "HEAD")
    upstream = git(repo, "rev-parse", "@{upstream}")
    if head != upstream:
        raise RuntimeError(f"HEAD is {head[:7]} but the remote is {upstream[:7]}. Push, then pull on Colab before running.")
    return head

def check_tag(repo: Path, tag: str) -> None:
    """
    Fails unless a tag is on the remote, so a run that must come after a timestamped commit can't run before it.

    Args:
        repo (Path): Path of the repo.
        tag (str): Name of the tag.
    Returns:
        None.
    """
    if not git(repo, "ls-remote", "--tags", "origin", f"refs/tags/{tag}"):
        raise RuntimeError(f"The tag {tag} isn't on the remote. Push it before running this section.")
