"""VI.3 - Answer generation with Qwen/Qwen3-0.6B."""

from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Dict, List

from src.models.index_build import IndexReport
from src.repositories.corpus_repository import corpus_repository
from src.schemas import MinimalAnswer, MinimalSource
from src.services.generator import (AnswerGenerator, DEFAULT_MODEL,
                                    NO_CONTEXT_ANSWER, SYSTEM_PROMPT)

from .helpers import (MD_LORA, QWEN, CLIResult, GenerateCalls, MiniCorpus,
                      fake_answer, requires_model)

LORA_MD = "data/raw/proj/docs/features/lora.md"


class WordTokenizer:
    """A tokenizer whose tokens are whitespace-separated words."""

    eos_token_id = 0

    def __init__(self) -> None:
        """Record the chat-template calls."""
        self.template_kwargs: Dict[str, Any] = {}
        self.messages: List[Dict[str, str]] = []

    def encode(self, text: str) -> List[str]:
        """Split on whitespace."""
        return text.split()

    def decode(self, ids: List[str],
               skip_special_tokens: bool = False) -> str:
        """Join words."""
        return " ".join(ids)

    def apply_chat_template(self, messages: List[Dict[str, str]],
                            **kwargs: Any) -> str:
        """Remember the prompt and its options."""
        self.messages = messages
        self.template_kwargs = kwargs
        return "PROMPT"

    def __call__(self, prompt: str, return_tensors: str) -> "FakeInputs":
        """Return three fake prompt tokens."""
        return FakeInputs(input_ids=FakeIds())


class FakeIds:
    """Prompt token ids of shape (1, 3)."""

    shape = (1, 3)


class FakeInputs(Dict[str, Any]):
    """Tokenizer output that can be moved to a device."""

    def to(self, device: str) -> "FakeInputs":
        """No-op device move."""
        return self


class FakeModel:
    """A model that 'generates' a thinking block and an answer."""

    def __init__(self) -> None:
        """Record generation options."""
        self.kwargs: Dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> List[List[str]]:
        """Return the prompt ids followed by new tokens."""
        self.kwargs = kwargs
        return [["p", "p", "p", "<think>", "</think>", "Use", "the",
                 "/v1/load_lora_adapter", "endpoint."]]


class FakeTorch:
    """Just enough of torch for ``answer``."""

    @staticmethod
    def no_grad() -> ContextManager[None]:
        """Disable nothing."""
        return nullcontext()


def make_generator(budget: int = 3000) -> AnswerGenerator:
    """An AnswerGenerator wired to fakes (no weights loaded)."""
    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.tokenizer = WordTokenizer()
    generator.model = FakeModel()
    generator.torch = FakeTorch()
    generator.device = "cpu"
    generator.corpus = corpus_repository
    generator.max_context_tokens = budget
    generator.max_new_tokens = 64
    return generator


def whole_file(path: str, length: int) -> MinimalSource:
    """A source covering ``length`` characters of a file."""
    return MinimalSource(file_path=path, first_character_index=0,
                         last_character_index=length)


def test_context_contains_numbered_sources(mini_corpus: MiniCorpus) -> None:
    """Each snippet is labelled with its rank and file path."""
    generator = make_generator()
    context = generator.build_context([whole_file(LORA_MD, len(MD_LORA)),
                                       whole_file("data/raw/proj/notes.txt",
                                                  20)])
    assert context.startswith(f"[1] {LORA_MD}\n# LoRA Adapters")
    assert "[2] data/raw/proj/notes.txt" in context


def test_context_respects_the_token_budget(mini_corpus: MiniCorpus) -> None:
    """Snippets are added only while they fit the token budget."""
    generator = make_generator(budget=40)
    sources = [whole_file(LORA_MD, len(MD_LORA))] * 5
    context = generator.build_context(sources)
    assert 0 < len(context.split()) <= 40


def test_unreadable_sources_are_skipped(mini_corpus: MiniCorpus) -> None:
    """A source pointing to a missing file does not break generation."""
    generator = make_generator()
    missing = whole_file("data/raw/proj/missing.md", 10)
    context = generator.build_context([missing,
                                       whole_file(LORA_MD, len(MD_LORA))])
    assert "[2] " + LORA_MD in context and "missing.md" not in context
    assert generator.answer("q", [missing]) == NO_CONTEXT_ANSWER


def test_prompt_strategy_is_grounded_and_deterministic(
        mini_corpus: MiniCorpus) -> None:
    """The model sees the context, must stay grounded, decodes greedily."""
    generator = make_generator()
    answer = generator.answer("Which endpoint loads a LoRA adapter?",
                              [whole_file(LORA_MD, len(MD_LORA))])
    tokenizer = generator.tokenizer
    model = generator.model
    assert isinstance(tokenizer, WordTokenizer)
    assert isinstance(model, FakeModel)
    system, user = tokenizer.messages
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert "ONLY from the provided context" in SYSTEM_PROMPT
    assert "/v1/load_lora_adapter" in user["content"]
    assert user["content"].endswith(
        "Question: Which endpoint loads a LoRA adapter?")
    assert tokenizer.template_kwargs["enable_thinking"] is False
    assert model.kwargs["do_sample"] is False
    assert answer == "Use the /v1/load_lora_adapter endpoint."


def test_answer_command_outputs_a_minimal_answer(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_generator: GenerateCalls,
        cli: Callable[..., CLIResult]) -> None:
    """``answer <query> --k`` prints MinimalAnswer JSON."""
    result = cli("answer", "load a LoRA adapter at runtime", "--k", "2")
    assert result.code == 0, result.err
    parsed = MinimalAnswer.model_validate_json(result.out)
    assert parsed.answer == fake_answer(parsed.question,
                                        parsed.retrieved_sources)
    assert fake_generator.models == [DEFAULT_MODEL]


def test_no_model_is_loaded_when_nothing_is_retrieved(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_generator: GenerateCalls,
        cli: Callable[..., CLIResult]) -> None:
    """A nonsensical question is answered without loading the LLM."""
    result = cli("answer", "xqzzv blorptang", "--k", "3")
    assert result.code == 0, result.err
    assert MinimalAnswer.model_validate_json(result.out).answer == \
        NO_CONTEXT_ANSWER
    assert fake_generator.loads == 0


@requires_model(QWEN)
def test_qwen_answers_from_the_retrieved_context(
        mini_corpus: MiniCorpus) -> None:
    """The real Qwen3-0.6B produces a coherent, grounded answer."""
    generator = AnswerGenerator(QWEN, max_new_tokens=96)
    answer = generator.answer(
        "Which HTTP endpoint loads a LoRA adapter at runtime?",
        [whole_file(LORA_MD, len(MD_LORA))])
    assert answer and "<think>" not in answer
    assert "load_lora_adapter" in answer
