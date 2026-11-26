"""Chapter V - Common Instructions (general rules, Makefile, extras)."""

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest
from pydantic import BaseModel

from src import schemas as models

from .helpers import PROJECT_ROOT

SRC = PROJECT_ROOT / "src"
LINT_FLAGS = ("--warn-return-any --warn-unused-ignores "
              "--ignore-missing-imports --disallow-untyped-defs "
              "--check-untyped-defs")


def source_files() -> List[Path]:
    """Every Python file of the project package."""
    return sorted(SRC.rglob("*.py"))


def parsed_sources() -> Iterator[Tuple[Path, ast.Module]]:
    """``(path, syntax tree)`` of every project file."""
    for path in source_files():
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def functions_and_classes(tree: ast.AST) -> Iterator[ast.AST]:
    """Every function, method and class definition of a tree."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            yield node


# ---------------------------------------------------------------- V.1 rules

def test_python_310_or_later() -> None:
    """The project declares and runs on Python >= 3.10."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.10"' in pyproject
    assert sys.version_info >= (3, 10)


def test_flake8_passes() -> None:
    """The whole repository follows flake8."""
    result = subprocess.run([sys.executable, "-m", "flake8", "."],
                            cwd=PROJECT_ROOT, capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_mypy_passes_with_the_mandatory_flags() -> None:
    """mypy reports no error with the flags of the ``lint`` rule."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", ".", *LINT_FLAGS.split()],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_function_and_class_has_a_docstring() -> None:
    """PEP 257: modules, classes and functions are documented."""
    missing = []
    for path, tree in parsed_sources():
        if ast.get_docstring(tree) is None:
            missing.append(f"{path.name}: module")
        for node in functions_and_classes(tree):
            assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef))
            if ast.get_docstring(node) is None:
                missing.append(f"{path.name}:{node.lineno} {node.name}")
    assert not missing, missing


def test_every_function_has_type_hints() -> None:
    """Parameters and return values are annotated everywhere."""
    untyped = []
    for path, tree in parsed_sources():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args.posonlyargs + node.args.args \
                + node.args.kwonlyargs
            names = [a.arg for a in args if a.annotation is None
                     and a.arg not in ("self", "cls")]
            for extra in (node.args.vararg, node.args.kwarg):
                if extra is not None and extra.annotation is None:
                    names.append(extra.arg)
            if node.returns is None:
                names.append("return")
            if names:
                untyped.append(f"{path.name}:{node.lineno} {names}")
    assert not untyped, untyped


def test_files_are_opened_with_context_managers() -> None:
    """Every ``open(...)`` is the subject of a ``with`` statement."""
    bare = []
    for path, tree in parsed_sources():
        managed = {id(item.context_expr) for node in ast.walk(tree)
                   if isinstance(node, ast.With) for item in node.items}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and \
                    isinstance(node.func, ast.Name) and \
                    node.func.id == "open" and id(node) not in managed:
                bare.append(f"{path.name}:{node.lineno}")
    assert not bare, bare


def test_sqlite_connections_are_closed() -> None:
    """Database connections are wrapped in ``closing(...)``."""
    text = (SRC / "bonus/cache/cache_repository.py").read_text()
    uses = re.findall(r"with closing\(self\._connect\(\)\)", text)
    assert len(uses) == text.count("self._connect()") >= 2


def test_cli_entry_point_catches_every_exception() -> None:
    """Errors are turned into messages instead of tracebacks."""
    tree = ast.parse((SRC / "views" / "cli.py").read_text())
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    handlers = [h for n in ast.walk(main) if isinstance(n, ast.Try)
                for h in n.handlers]
    caught = {ast.unparse(h.type) for h in handlers if h.type is not None}
    assert {"RagError", "Exception", "KeyboardInterrupt"} <= caught


# ------------------------------------------------------------ V.2 Makefile

MAKEFILE = (PROJECT_ROOT / "Makefile").read_text()


@pytest.mark.parametrize("rule", ["install", "run", "debug", "clean", "lint",
                                  "lint-strict"])
def test_makefile_has_rule(rule: str) -> None:
    """The mandatory (and the optional lint-strict) rules exist."""
    assert re.search(rf"^{re.escape(rule)}:", MAKEFILE, re.MULTILINE)


def make_dry_run(rule: str) -> str:
    """Commands ``make`` would run for ``rule``, without running them."""
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    result = subprocess.run(["make", "-n", rule], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return " ".join(result.stdout.replace("\\\n", " ").split())


def test_make_install_uses_uv() -> None:
    """``install`` installs the dependencies with uv."""
    assert "uv sync" in make_dry_run("install")


def test_make_lint_runs_flake8_and_mypy_with_the_mandatory_flags() -> None:
    """``lint`` runs flake8 and mypy with exactly the required flags."""
    commands = make_dry_run("lint")
    assert "flake8 ." in commands
    assert f"mypy . {LINT_FLAGS}" in commands


def test_make_lint_strict_runs_mypy_strict() -> None:
    """``lint-strict`` runs flake8 and mypy --strict."""
    commands = make_dry_run("lint-strict")
    assert "flake8 ." in commands and "mypy . --strict" in commands


def test_make_debug_uses_pdb() -> None:
    """``debug`` runs the program under Python's debugger."""
    assert "-m pdb -m src" in make_dry_run("debug")


def test_make_run_executes_the_program() -> None:
    """``run`` executes the main module."""
    assert "-m src" in make_dry_run("run")


def test_make_clean_removes_caches() -> None:
    """``clean`` removes __pycache__ and .mypy_cache."""
    commands = make_dry_run("clean")
    assert "__pycache__" in commands and ".mypy_cache" in commands


# ------------------------------------------------- V.3 additional guidelines

def test_gitignore_excludes_python_artifacts() -> None:
    """A .gitignore keeps caches, venvs and data out of git."""
    ignored = (PROJECT_ROOT / ".gitignore").read_text().split()
    for pattern in ("__pycache__/", ".venv/", ".mypy_cache/", "data/"):
        assert pattern in ignored


def test_tests_use_pytest() -> None:
    """Unit tests run with pytest (declared as a dev dependency)."""
    assert '"pytest' in (PROJECT_ROOT / "pyproject.toml").read_text()


# ----------------------------------------------- V.4 additional requirements

@pytest.mark.parametrize("name", [
    "MinimalSource", "UnansweredQuestion", "AnsweredQuestion", "RagDataset",
    "MinimalSearchResults", "MinimalAnswer", "StudentSearchResults",
    "StudentSearchResultsAndAnswer"])
def test_data_models_are_pydantic(name: str) -> None:
    """Data exchanged between stages is validated by pydantic."""
    assert issubclass(getattr(models, name), BaseModel)


def test_project_is_managed_by_uv() -> None:
    """pyproject.toml and uv.lock sit at the repository root."""
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "uv.lock").is_file()


@pytest.mark.parametrize("package", ["fire", "pydantic", "tqdm"])
def test_required_libraries_are_dependencies(package: str) -> None:
    """Fire, pydantic and tqdm are declared dependencies."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    assert re.search(rf'"{package}[><=~]', pyproject)


def test_cli_is_built_with_python_fire() -> None:
    """The entry point hands the CLI class to ``fire.Fire``."""
    text = (SRC / "views" / "cli.py").read_text()
    assert "fire.Fire(RagCLI)" in text


@pytest.mark.parametrize("module", ["services/rag_service.py",
                                    "services/indexer.py",
                                    "bonus/semantic/embedder.py",
                                    "extra/langchain/indexer.py"])
def test_long_operations_show_tqdm_progress(module: str) -> None:
    """Indexing, searching, embedding and answering use tqdm."""
    assert "tqdm(" in (SRC / module).read_text()


def test_default_model_is_qwen3_0_6b() -> None:
    """Answers are generated with Qwen/Qwen3-0.6B by default."""
    from src.config import settings
    from src.services.generator import DEFAULT_MODEL

    assert settings.llm_model == DEFAULT_MODEL == "Qwen/Qwen3-0.6B"


def test_flake8_also_covers_bonus_files() -> None:
    """Bonus modules are linted like the rest (src is never excluded)."""
    config = (PROJECT_ROOT / ".flake8").read_text()
    excluded = re.search(r"^exclude\s*=\s*(.*)$", config, re.MULTILINE)
    assert excluded is not None
    assert not any(part.strip().startswith(("src", "./src"))
                   for part in excluded.group(1).split(","))
