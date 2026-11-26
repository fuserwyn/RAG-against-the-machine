"""LangChain ingestion: split the corpus and persist a BM25Retriever."""

from typing import Any, List

from tqdm import tqdm

from ...repositories.corpus_repository import (CorpusRepository,
                                               corpus_repository)
from ...repositories.meta_repository import MetaRepository, meta_repository
from ...services.tokenizer import Tokenizer, tokenizer
from .chunker import LangChainChunker, langchain_chunker
from .langchain_repository import LangChainRepository, langchain_repository


class LangChainIndexer:
    """Builds the LangChain index (splitters + ``BM25Retriever``)."""

    def __init__(self, corpus: CorpusRepository, store: LangChainRepository,
                 meta: MetaRepository, chunker: LangChainChunker,
                 text_tokenizer: Tokenizer) -> None:
        """Wire the repositories and services the build relies on."""
        self.corpus = corpus
        self.store = store
        self.meta = meta
        self.chunker = chunker
        self.tokenizer = text_tokenizer

    def build(self, raw_dir: str, index_dir: str,
              max_chunk_size: int) -> int:
        """Chunk with LangChain splitters and persist a BM25Retriever.

        Returns:
            The number of chunks indexed.
        """
        from langchain_community.retrievers import BM25Retriever

        files = self.corpus.discover(raw_dir)
        if not files:
            raise FileNotFoundError(
                f"no indexable files found in {raw_dir!r}")
        documents = self._documents(files, max_chunk_size)
        retriever = BM25Retriever.from_documents(
            documents, preprocess_func=self.tokenizer.tokenize)
        self.store.save(index_dir, retriever)
        self.meta.write(index_dir, raw_dir=raw_dir,
                        max_chunk_size=max_chunk_size, n_files=len(files),
                        n_chunks=len(documents))
        return len(documents)

    def _documents(self, files: List[str], max_chunk_size: int) -> List[Any]:
        """One LangChain ``Document`` per chunk, with its span as metadata."""
        from langchain_core.documents import Document

        splitters = self.chunker.make_splitters(max_chunk_size)
        documents: List[Any] = []
        for path in tqdm(files, desc="Chunking files (langchain)",
                         unit="file"):
            try:
                text = self.corpus.read_text(path)
            except OSError as err:
                tqdm.write(f"Skipping {path}: {err}")
                continue
            path_words = path.replace("/", " ").replace(".", " ")
            splitter = splitters[self.chunker.language_of(path)]
            for start, end in self.chunker.split_spans(splitter, text,
                                                       max_chunk_size):
                body = text[start:end]
                if body.strip():
                    documents.append(Document(
                        page_content=f"{path_words}\n{body}",
                        metadata={"file_path": path,
                                  "first_character_index": start,
                                  "last_character_index": end}))
        return documents


langchain_indexer = LangChainIndexer(
    corpus=corpus_repository, store=langchain_repository,
    meta=meta_repository, chunker=langchain_chunker,
    text_tokenizer=tokenizer)
