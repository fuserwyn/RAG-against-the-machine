"""X - Submission: repository content and what must stay out of it."""

import shutil
import subprocess

import pytest

from .helpers import PROJECT_ROOT


def git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command in the repository."""
    if shutil.which("git") is None or \
            not (PROJECT_ROOT / ".git").exists():
        pytest.skip("not a git repository")
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True,
                          capture_output=True, timeout=60)


@pytest.mark.parametrize("path", ["src", "pyproject.toml", "uv.lock",
                                  "Makefile", "README.md", ".gitignore"])
def test_required_content(path: str) -> None:
    """src/, uv files, Makefile and README are in the repository."""
    assert (PROJECT_ROOT / path).exists()


@pytest.mark.parametrize("path", [
    "data/raw/vllm-0.10.1/setup.py", "data/processed/native/chunks.jsonl",
    "data/output/search_results/x.json", "moulinette.zip",
    "vllm-0.10.1.zip", "moulinette-ubuntu", ".venv/bin/python",
    "src/__pycache__/cli.cpython-312.pyc"])
def test_large_or_generated_files_are_ignored(path: str) -> None:
    """Corpus, indexes, outputs, archives and caches are git-ignored."""
    assert git("check-ignore", "-q", path).returncode == 0, path


def test_no_big_file_would_be_committed() -> None:
    """Every file git would add is smaller than 2 MB."""
    listing = git("ls-files", "--cached", "--others", "--exclude-standard")
    for name in listing.stdout.splitlines():
        path = PROJECT_ROOT / name
        if path.is_file():
            assert path.stat().st_size < 2_000_000, name


def test_no_model_weights_in_the_repository() -> None:
    """No weight files outside the ignored folders."""
    listing = git("ls-files", "--cached", "--others", "--exclude-standard")
    for name in listing.stdout.splitlines():
        assert not name.endswith((".safetensors", ".bin", ".pt", ".npy",
                                  ".npz", ".pkl")), name
