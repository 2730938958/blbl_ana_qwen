from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from .schema import CommentDoc


DEFAULT_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def docs_to_lc_documents(docs: Sequence[CommentDoc]) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        meta = asdict(d)
        # avoid duplicating long text in metadata
        meta.pop("comment_text", None)
        meta.pop("clean_text", None)
        out.append(Document(page_content=d.clean_text, metadata=meta))
    return out


def build_faiss(
    docs: Sequence[CommentDoc],
    *,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    normalize_embeddings: bool = True,
) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )
    lc_docs = docs_to_lc_documents(docs)
    return FAISS.from_documents(lc_docs, embeddings)


def save_faiss(store: FAISS, dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)
    store.save_local(dir_path)


def load_faiss(
    dir_path: str,
    *,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    normalize_embeddings: bool = True,
) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )
    return FAISS.load_local(dir_path, embeddings, allow_dangerous_deserialization=True)


def write_docs_jsonl(docs: Iterable[CommentDoc], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def read_docs_jsonl(path: str) -> List[CommentDoc]:
    out: List[CommentDoc] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out.append(CommentDoc(**obj))
    return out


def search_comments(
    store: FAISS,
    query: str,
    *,
    k: int = 6,
) -> List[Tuple[Document, float]]:
    return store.similarity_search_with_score(query, k=k)

