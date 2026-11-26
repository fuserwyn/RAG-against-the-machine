"""Services: the logic of the mandatory pipeline.

chunk -> tokenize -> BM25 index -> retrieve -> answer -> evaluate.
The bonuses live in ``src/bonus``, the extra backends in ``src/extra``;
``rag_service`` and ``retriever_factory`` plug them in when asked for.
Each module ends with a ready-to-use instance (``tokenizer``,
``file_chunker``, ``native_indexer``, ...). Classes that need an index
directory or model weights (retrievers, ``RagService``) are built by the
caller instead.
"""
