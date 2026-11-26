"""Errors shown to the user as a message, never as a traceback."""


class RagError(Exception):
    """A user-facing error (bad argument, missing file, missing index)."""
