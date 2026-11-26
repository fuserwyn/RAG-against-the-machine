"""Repositories: every read and write of a file, in one layer.

Services never open a file themselves; they ask a repository. Each module
ends with a ready-to-use instance (``corpus_repository``, ...).
"""
