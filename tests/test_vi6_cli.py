"""VI.6 - Command-Line Interface: commands, options and degenerate input."""

import inspect
import json
import os
import subprocess
import sys
from typing import Callable, List

import pytest

from src.models.index_build import IndexReport
from src.views.cli import RagCLI

from .helpers import PROJECT_ROOT, CLIResult, GenerateCalls, MiniCorpus

REQUIRED = {
    "index": ["max_chunk_size"],
    "search": ["query", "k"],
    "search_dataset": ["dataset_path", "k", "save_directory"],
    "answer": ["query", "k"],
    "answer_dataset": ["student_search_results_path", "save_directory"],
    "evaluate": ["student_search_results_path", "dataset_path"],
}


@pytest.mark.parametrize("command", sorted(REQUIRED))
def test_required_commands_and_options_exist(command: str) -> None:
    """Every mandatory command exists with its mandatory options."""
    method = getattr(RagCLI, command)
    parameters = inspect.signature(method).parameters
    for option in REQUIRED[command]:
        assert option in parameters, f"{command} lacks --{option}"


def test_help_lists_the_commands(cli: Callable[..., CLIResult]) -> None:
    """Running without a command shows the available commands."""
    result = cli()
    text = result.out + result.err
    for command in REQUIRED:
        assert command in text


def assert_clean_error(result: CLIResult, *fragments: str) -> None:
    """Exit code 1, an ``Error:`` last line, no traceback."""
    assert result.code == 1, (result.out, result.err)
    assert result.err.strip().splitlines()[-1].startswith("Error: ")
    assert "Traceback" not in result.err + result.out
    for fragment in fragments:
        assert fragment in result.err


@pytest.mark.parametrize("command", ["search", "answer"])
@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query(mini_corpus: MiniCorpus, mini_index: IndexReport,
                     cli: Callable[..., CLIResult], command: str,
                     query: str) -> None:
    """An empty query is refused cleanly."""
    assert_clean_error(cli(command, query), "the query is empty")


def test_nonsensical_query_returns_no_sources(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        cli: Callable[..., CLIResult]) -> None:
    """Gibberish is valid input: it simply matches nothing."""
    result = cli("search", "qwxzzt vlorp ##!!", "--k", "5")
    assert result.code == 0, result.err
    assert json.loads(result.out)["retrieved_sources"] == []


def test_numeric_query_is_accepted(mini_corpus: MiniCorpus,
                                   mini_index: IndexReport,
                                   cli: Callable[..., CLIResult]) -> None:
    """Fire parses ``42`` as an int; it is still searched as text."""
    result = cli("search", "42")
    assert result.code == 0, result.err
    assert json.loads(result.out)["question"] == "42"


@pytest.mark.parametrize("command", ["search", "answer"])
@pytest.mark.parametrize("k,message", [
    ("0", "k must be positive"), ("-3", "k must be positive"),
    ("abc", "k must be an integer"), ("2.5", "k must be an integer"),
    ("True", "k must be an integer")])
def test_invalid_k(mini_corpus: MiniCorpus, mini_index: IndexReport,
                   cli: Callable[..., CLIResult], command: str, k: str,
                   message: str) -> None:
    """k = 0, negative or non-integer k are refused cleanly."""
    assert_clean_error(cli(command, "lora", "--k", k), message)


def test_invalid_k_for_search_dataset(mini_corpus: MiniCorpus,
                                      mini_index: IndexReport,
                                      cli: Callable[..., CLIResult]) -> None:
    """search_dataset validates k before touching the dataset."""
    assert_clean_error(cli("search_dataset", "--dataset_path", "x.json",
                           "--k", "0"), "k must be positive")


MISSING_FILE_COMMANDS: List[List[str]] = [
    ["search_dataset", "--dataset_path", "missing.json"],
    ["answer_dataset", "--student_search_results_path", "missing.json"],
    ["evaluate", "--student_search_results_path", "missing.json",
     "--dataset_path", "missing.json"],
]


@pytest.mark.parametrize("args", MISSING_FILE_COMMANDS)
def test_missing_files(mini_corpus: MiniCorpus,
                       cli: Callable[..., CLIResult],
                       args: List[str]) -> None:
    """A missing input file is reported, not raised."""
    assert_clean_error(cli(*args), "file not found: missing.json")


@pytest.mark.parametrize("content", ["{bad json", "", "[1, 2", "\xff\xfe"])
@pytest.mark.parametrize("command,flag", [
    ("search_dataset", "--dataset_path"),
    ("answer_dataset", "--student_search_results_path")])
def test_malformed_json(mini_corpus: MiniCorpus, mini_index: IndexReport,
                        cli: Callable[..., CLIResult], content: str,
                        command: str, flag: str) -> None:
    """Malformed JSON is reported, not raised."""
    path = mini_corpus.root / "bad.json"
    path.write_text(content, encoding="latin-1")
    assert_clean_error(cli(command, flag, str(path)), "malformed JSON")


@pytest.mark.parametrize("payload", [{"questions": []},
                                     {"rag_questions": [{"id": 1}]},
                                     [1, 2, 3]])
def test_json_with_the_wrong_schema(mini_corpus: MiniCorpus,
                                    mini_index: IndexReport,
                                    cli: Callable[..., CLIResult],
                                    payload: object) -> None:
    """Valid JSON that is not a dataset is reported cleanly."""
    path = mini_corpus.root / "wrong.json"
    path.write_text(json.dumps(payload))
    assert_clean_error(cli("search_dataset", "--dataset_path", str(path)),
                       "invalid dataset")
    assert_clean_error(cli("answer_dataset", "--student_search_results_path",
                           str(path)), "invalid search results")
    assert_clean_error(cli("evaluate", "--student_search_results_path",
                           str(path), "--dataset_path", str(path)),
                       "invalid search results")


def test_evaluate_needs_answered_questions(
        mini_corpus: MiniCorpus, cli: Callable[..., CLIResult]) -> None:
    """Evaluating against an unanswered dataset is an explicit error."""
    results = mini_corpus.root / "results.json"
    results.write_text(json.dumps({"search_results": [], "k": 5}))
    dataset = mini_corpus.root / "dataset.json"
    dataset.write_text(json.dumps({"rag_questions": [{"question": "?"}]}))
    assert_clean_error(cli("evaluate", "--student_search_results_path",
                           str(results), "--dataset_path", str(dataset)),
                       "no answered questions")


def test_missing_index(mini_corpus: MiniCorpus,
                       cli: Callable[..., CLIResult]) -> None:
    """Searching before indexing tells the user to run ``index``."""
    assert_clean_error(cli("search", "lora"), "run the 'index' command")


@pytest.mark.parametrize("args,message", [
    (["--backend", "elastic", "search", "x"], "unknown backend"),
    (["--mode", "magic", "search", "x"], "unknown mode"),
    (["--backend", "langchain", "--mode", "semantic", "search", "x"],
     "only available with the native backend"),
    (["--cache", "maybe", "search", "x"], "cache must be True or False"),
    (["index", "--semantic", "yes"], "semantic must be True or False"),
])
def test_invalid_global_options(mini_corpus: MiniCorpus,
                                cli: Callable[..., CLIResult],
                                args: List[str], message: str) -> None:
    """Invalid options are refused with a clear message."""
    assert_clean_error(cli(*args), message)


def test_unwritable_save_directory(mini_corpus: MiniCorpus,
                                   mini_index: IndexReport,
                                   cli: Callable[..., CLIResult]) -> None:
    """A save directory that cannot be created is reported."""
    (mini_corpus.root / "a_file").write_text("x")
    dataset = mini_corpus.root / "d.json"
    dataset.write_text(json.dumps({"rag_questions": [{"question": "q"}]}))
    result = cli("search_dataset", "--dataset_path", str(dataset),
                 "--save_directory", "a_file/sub")
    assert_clean_error(result, "cannot write into")
    assert "Searching" not in result.err, "must fail before the work"


def test_corrupted_index(mini_corpus: MiniCorpus, mini_index: IndexReport,
                         cli: Callable[..., CLIResult]) -> None:
    """A damaged index file is reported, not raised."""
    with open(os.path.join(mini_corpus.native_dir, "bm25_vocab.json"),
              "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert_clean_error(cli("search", "lora"), "corrupted index")


def test_missing_argument_is_a_fire_usage_error(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        cli: Callable[..., CLIResult]) -> None:
    """Forgetting the query prints Fire's usage, not a traceback."""
    result = cli("search")
    assert result.code != 0 and "Traceback" not in result.err
    assert "query" in result.err


def test_paths_are_cli_arguments(mini_corpus: MiniCorpus,
                                 cli: Callable[..., CLIResult]) -> None:
    """Corpus and index locations are options, not hard-coded paths."""
    os.rename(mini_corpus.root / "data/raw", mini_corpus.root / "corpus")
    result = cli("--index_dir", "elsewhere", "index", "--raw_dir", "corpus")
    assert result.code == 0, result.err
    assert (mini_corpus.root / "elsewhere/native/chunks.jsonl").is_file()
    found = cli("--index_dir", "elsewhere", "search", "zephyrquokka")
    assert '"file_path": "corpus/proj/docs/zephyr.md"' in found.out


def test_answer_dataset_accepts_a_model_name(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_generator: GenerateCalls,
        cli: Callable[..., CLIResult]) -> None:
    """The generation model can be changed from the CLI."""
    cli("answer", "lora adapter", "--model_name", "my/model")
    assert fake_generator.models == ["my/model"]


def test_python_dash_m_src_never_prints_a_traceback(
        mini_corpus: MiniCorpus) -> None:
    """End to end through ``python -m src``: clean errors only."""
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    for args in (["search", ""], ["search", "x", "--k", "0"],
                 ["search_dataset", "--dataset_path", "nope.json"],
                 ["evaluate", "--student_search_results_path", "x",
                  "--dataset_path", "y"],
                 ["search", "lora"]):
        result = subprocess.run([sys.executable, "-m", "src", *args],
                                cwd=mini_corpus.root, env=env, text=True,
                                capture_output=True, timeout=120)
        assert result.returncode == 1, args
        assert result.stderr.startswith("Error: "), result.stderr
        assert "Traceback" not in result.stderr
