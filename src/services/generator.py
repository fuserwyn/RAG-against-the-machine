"""Answer generation with a small local causal language model (Qwen3)."""

from typing import Any, Callable, Dict, List

from ..bonus.semantic.embedder import pick_device
from ..config import settings
from ..repositories.corpus_repository import (CorpusRepository,
                                              corpus_repository)
from ..schemas.source import MinimalSource

DEFAULT_MODEL = settings.llm_model
SYSTEM_PROMPT = (
    "You are an assistant answering questions about the vLLM codebase. "
    "Answer ONLY from the provided context snippets. Be concise (2-4 "
    "sentences), quote exact names of functions, classes, parameters or "
    "endpoints when relevant. If the context does not contain the answer, "
    "say that the retrieved context does not contain it."
)
NO_CONTEXT_ANSWER = "No relevant context was retrieved for this question."


class AnswerGenerator:
    """Wraps a Hugging Face model to answer from retrieved text."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 max_context_tokens: int = settings.max_context_tokens,
                 max_new_tokens: int = settings.max_new_tokens,
                 corpus: CorpusRepository = corpus_repository) -> None:
        """Load the tokenizer and model (lazily imports torch).

        Args:
            model_name: Hugging Face model id.
            max_context_tokens: Token budget for the retrieved context.
            max_new_tokens: Maximum length of the generated answer.
            corpus: Where the source snippets are read from.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch: Any = torch
        self.device = pick_device(torch)
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto")
        self.model: Any = model.to(self.device)
        self.model.eval()
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.corpus = corpus

    def build_context(self, sources: List[MinimalSource]) -> str:
        """Concatenate source snippets until the token budget is reached."""
        parts: List[str] = []
        used = 0
        for number, source in enumerate(sources, start=1):
            try:
                snippet = self.corpus.read_span(source)
            except OSError:
                continue
            block = f"[{number}] {source.file_path}\n{snippet.strip()}\n"
            n_tokens = len(self.tokenizer.encode(block))
            if used + n_tokens > self.max_context_tokens:
                if not parts:               # first block alone is too big
                    ids = self.tokenizer.encode(block)
                    parts.append(self.tokenizer.decode(
                        ids[:self.max_context_tokens]))
                break
            parts.append(block)
            used += n_tokens
        return "\n".join(parts)

    def messages(self, question: str, context: str) -> List[Dict[str, str]]:
        """The chat given to the model: rules, then context + question."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

    def answer(self, question: str, sources: List[MinimalSource]) -> str:
        """Generate a grounded answer for ``question``."""
        context = self.build_context(sources)
        if not context:
            return NO_CONTEXT_ANSWER
        prompt = self.tokenizer.apply_chat_template(
            self.messages(question, context), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text: str = self.tokenizer.decode(new_tokens,
                                          skip_special_tokens=True)
        return text.replace("<think>", "").replace("</think>", "").strip()


class GeneratorProvider:
    """Creates generators; tests swap ``factory`` for an instant fake."""

    def __init__(self,
                 factory: Callable[[str], Any] = AnswerGenerator) -> None:
        """Remember how to build a generator."""
        self.factory = factory

    def create(self, model_name: str = DEFAULT_MODEL) -> Any:
        """A new generator for ``model_name`` (loads the weights)."""
        return self.factory(model_name)


generator_provider = GeneratorProvider()
