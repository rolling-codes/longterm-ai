"""Retrieval engine: index accumulated evidence and pull back the relevant slice.

The "retrieval-first" piece of the spec. In the run loop every tool result is bounded
to a summary while its full text lands in the :class:`~longterm_ai.tools.ArtifactStore`
and is never read again. This module indexes that evidence (BM25, stdlib only) so a later
turn can retrieve the top matches under a token budget instead of re-dumping raw output.

Keyword retrieval, no embeddings and no vector backend -- honest and dependency-free.
# ponytail: BM25 keyword retrieval, swap in embeddings when a vector backend lands.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .contracts import RiskClass, ToolContext, ToolDefinition
from .telemetry import rough_tokens
from .tools import FunctionTool

_WORD = re.compile(r"\w+")   # keeps identifiers whole (module_7), good for code/log evidence

# Okapi BM25 constants (standard defaults).
_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Shared by indexing and querying so they match."""
    return _WORD.findall(text.lower())


@dataclass
class Document:
    """One indexed unit of evidence."""
    id: str
    text: str
    source: str = ""                               # artifact_ref / tool name / "memory"
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """Incremental Okapi BM25 over tokenized documents. Pure stdlib."""

    def __init__(self) -> None:
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)  # term -> {doc_id: tf}
        self._doc_len: dict[str, int] = {}
        self._total_len = 0

    @property
    def n_docs(self) -> int:
        return len(self._doc_len)

    def add(self, doc_id: str, tokens: list[str]) -> None:
        if doc_id in self._doc_len:            # re-index: remove the old contribution first
            self._remove(doc_id)
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for term, count in tf.items():
            self._postings[term][doc_id] = count
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)

    def _remove(self, doc_id: str) -> None:
        for term in list(self._postings):
            if doc_id in self._postings[term]:
                del self._postings[term][doc_id]
                if not self._postings[term]:
                    del self._postings[term]
        self._total_len -= self._doc_len.pop(doc_id, 0)

    def score(self, query_tokens: list[str]) -> dict[str, float]:
        """BM25 score per doc that matches at least one query term."""
        n = self.n_docs
        if n == 0:
            return {}
        avgdl = self._total_len / n
        scores: dict[str, float] = defaultdict(float)
        for term in query_tokens:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                dl = self._doc_len[doc_id]
                denom = tf + _K1 * (1 - _B + _B * dl / avgdl)
                scores[doc_id] += idf * (tf * (_K1 + 1)) / denom
        return dict(scores)


class MemoryStore:
    """In-memory evidence store: a BM25 index plus the documents themselves."""

    def __init__(self) -> None:
        self._index = BM25Index()
        self._docs: dict[str, Document] = {}
        self._order: dict[str, int] = {}       # insertion order, for stable tie-breaking
        self._seq = 0

    def index(self, doc: Document) -> None:
        if doc.id not in self._docs:
            self._order[doc.id] = self._seq
            self._seq += 1
        self._docs[doc.id] = doc
        self._index.add(doc.id, _tokenize(f"{doc.source} {doc.text}"))

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Top-k (Document, score) descending; ties broken by insertion order."""
        scores = self._index.score(_tokenize(query))
        ranked = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], self._order.get(kv[0], 0)),
        )
        return [(self._docs[doc_id], score) for doc_id, score in ranked[:k]]


class Retriever:
    """Thin query surface over a :class:`MemoryStore`, with an optional token budget."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(self, query: str, k: int = 5, token_budget: int | None = None) -> list[Document]:
        hits = [doc for doc, _ in self.store.search(query, k)]
        if token_budget is None:
            return hits
        kept: list[Document] = []
        used = 0
        for doc in hits:                       # drop from the tail once the budget is spent
            cost = rough_tokens(doc.text)
            if used + cost > token_budget:
                break
            kept.append(doc)
            used += cost
        return kept


def assemble_context(retriever: Retriever, query: str, budget: int, k: int = 5) -> str:
    """Retrieval-first context block: relevant evidence with provenance, under *budget*."""
    docs = retriever.retrieve(query, k=k, token_budget=budget)
    if not docs:
        return ""
    return "\n\n".join(f"[{d.source or d.id}]\n{d.text}" for d in docs)


def make_retrieve_tool(store: MemoryStore, *, budget: int = 1000, k: int = 5) -> FunctionTool:
    """A READ_ONLY tool the model calls through the gateway to search indexed evidence.

    The model proposes the query; the gateway authorizes and bounds the result, so
    retrieval rides the same fail-closed lifecycle as any other tool (REQ 002).
    """
    retriever = Retriever(store)

    def _run(args: dict[str, Any], ctx: ToolContext) -> str:
        return assemble_context(retriever, str(args["query"]), budget=budget, k=k)

    return FunctionTool(
        ToolDefinition(
            name="retrieve",
            arg_schema={"query": "str"},
            risk=RiskClass.READ_ONLY,
            idempotent=True,
            description="Search indexed task evidence and return the most relevant slices.",
        ),
        _run,
    )
