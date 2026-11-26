"""Validation of user arguments (CLI flags, HTTP parameters)."""

from typing import Any

from ..config import settings
from ..errors import RagError


class InputValidator:
    """Checks one argument and returns it, or raises :class:`RagError`."""

    def positive_int(self, value: Any, name: str) -> int:
        """An integer argument that must be > 0."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise RagError(f"{name} must be an integer, got {value!r}")
        if value <= 0:
            raise RagError(f"{name} must be positive, got {value}")
        return value

    def k(self, value: Any) -> int:
        """The number of results to return."""
        return self.positive_int(value, "k")

    def chunk_size(self, value: Any) -> int:
        """``--max_chunk_size``: positive and within the grader limit."""
        size = self.positive_int(value, "max_chunk_size")
        if size > settings.max_context_length:
            raise RagError("max_chunk_size must be <= "
                           f"{settings.max_context_length}")
        return size

    def flag(self, value: Any, name: str) -> bool:
        """A boolean flag (Fire gives ``True`` / ``False``)."""
        if value is not True and value is not False:
            raise RagError(f"{name} must be True or False, got {value!r}")
        return value

    def query(self, value: Any) -> str:
        """A free-text question that is not empty."""
        text = value if isinstance(value, str) else str(value)
        if not text.strip():
            raise RagError("the query is empty")
        return text

    def choice(self, value: Any, name: str, options: tuple[str, ...]) -> str:
        """One of a fixed set of strings."""
        if value not in options:
            raise RagError(f"unknown {name} {value!r}, "
                           f"choose one of {', '.join(options)}")
        return str(value)

    def port(self, value: Any) -> int:
        """A TCP port (0 lets the system pick one)."""
        if isinstance(value, bool) or not isinstance(value, int) \
                or not 0 <= value <= 65535:
            raise RagError(f"port must be an integer in 0..65535, got "
                           f"{value!r}")
        return value

    def text(self, value: Any, name: str) -> str:
        """A non-empty string."""
        if not isinstance(value, str) or not value:
            raise RagError(f"{name} must be a non-empty string")
        return value


validator = InputValidator()
