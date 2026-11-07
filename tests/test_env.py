import importlib.metadata as metadata
import subprocess

import pytest

from utils.env import check_pins, check_pushed, check_tag, read_pins

def run_git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args], check=True, capture_output=True)

@pytest.fixture
def repo(tmp_path):
    """A clone with a pushed commit, standing in for the repo on Colab."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(remote), str(clone)], check=True, capture_output=True)
    (clone / "file.txt").write_text("one")
    run_git(clone, "add", ".")
    run_git(clone, "commit", "-q", "-m", "first")
    run_git(clone, "push", "-q", "-u", "origin", "HEAD")
    return clone

def test_pushed_repo_passes(repo):
    assert len(check_pushed(repo)) == 40

def test_uncommitted_change_fails(repo):
    (repo / "file.txt").write_text("two")
    with pytest.raises(RuntimeError, match="uncommitted"):
        check_pushed(repo)

def test_untracked_file_fails(repo):
    (repo / "new.txt").write_text("new")
    with pytest.raises(RuntimeError, match="uncommitted"):
        check_pushed(repo)

def test_unpushed_commit_fails(repo):
    (repo / "file.txt").write_text("two")
    run_git(repo, "commit", "-q", "-am", "second")
    with pytest.raises(RuntimeError, match="Push"):
        check_pushed(repo)

def test_read_pins_removes_extras(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("torch==2.15.0\nlm-eval[hf,ifeval]==0.4.13\n")
    assert read_pins(path) == {"torch": "2.15.0", "lm-eval": "0.4.13"}

def test_check_pins(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text(f"pytest=={metadata.version('pytest')}\n")
    assert check_pins(path) == {"pytest": metadata.version("pytest")}
    path.write_text("pytest==0.0.1\n")
    with pytest.raises(RuntimeError):
        check_pins(path)

def test_requests_and_yaml_come_with_the_pinned_packages():
    requirements = " ".join(metadata.requires("wandb")).lower()
    assert "requests" in requirements
    assert "pyyaml" in requirements

def test_check_tag_fails_until_the_tag_is_on_the_remote(tmp_path):
    remote = tmp_path / "remote.git"
    local = tmp_path / "local"
    run = lambda *args, cwd=None: subprocess.run(args, cwd=cwd, check=True, capture_output=True)
    run("git", "init", "--bare", str(remote))
    run("git", "clone", str(remote), str(local))
    run("git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "--allow-empty", "-m", "first", cwd=local)
    run("git", "tag", "prereg-draft", cwd=local)
    with pytest.raises(RuntimeError, match="prereg-draft"):
        check_tag(local, "prereg-draft")
    run("git", "push", "origin", "prereg-draft", cwd=local)
    check_tag(local, "prereg-draft")
