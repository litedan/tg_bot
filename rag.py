"""Векторный индекс базы знаний: E5-эмбеддинги, Chroma, cross-encoder rerank.

Индекс пересобирается автоматически, когда меняется содержимое knowledge.txt
или модель эмбеддингов. Факт актуальности фиксируется в manifest-файле рядом
с chroma.sqlite3.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

import config

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = config.CHROMA_DIR
KNOWLEDGE_FILE = config.KNOWLEDGE_FILE
COLLECTION_NAME = config.COLLECTION_NAME
EMBEDDING_MODEL = config.EMBEDDING_MODEL
RERANKER_MODEL = config.RERANKER_MODEL
RETRIEVAL_CANDIDATES = config.RETRIEVAL_CANDIDATES
RERANKED_DOCUMENTS = config.RERANKED_DOCUMENTS

MANIFEST_FILE = Path(CHROMA_DIR) / f"{COLLECTION_NAME}.manifest.json"

_embeddings = None
_reranker = None


class E5Embeddings(Embeddings):
    """Обёртка с обязательными query/passage префиксами для multilingual-e5."""

    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts):
        return self.model.embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text):
        return self.model.embed_query(f"query: {text}")


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = E5Embeddings()
    return _embeddings


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _reranker


def _load_chunks():
    if not KNOWLEDGE_FILE.is_file():
        raise FileNotFoundError(f"Файл базы знаний не найден: {KNOWLEDGE_FILE}")

    docs = TextLoader(str(KNOWLEDGE_FILE), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    print(f"База знаний разбита на {len(chunks)} чанков.")
    return chunks


def _fingerprint() -> dict:
    if not KNOWLEDGE_FILE.is_file():
        raise FileNotFoundError(f"Файл базы знаний не найден: {KNOWLEDGE_FILE}")
    data = KNOWLEDGE_FILE.read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "embedding_model": EMBEDDING_MODEL,
    }


def _read_manifest() -> dict:
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_manifest(fingerprint: dict, chunks: int) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "chunks": chunks,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = MANIFEST_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_FILE)


def _new_store() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def build_or_load_db(force_rebuild: bool = False) -> Chroma:
    """Загружает индекс или пересобирает его при изменении корпуса."""
    fingerprint = _fingerprint()
    manifest = _read_manifest()
    indexed = bool(manifest) and manifest.get("fingerprint") == fingerprint

    store = _new_store()
    count = store._collection.count()

    if force_rebuild or not indexed or count == 0:
        reason = "принудительно" if force_rebuild else "корпус изменился или индекс пуст"
        print(f"Пересборка индекса ({reason})...")
        try:
            store.delete_collection()
        except Exception:  # коллекция могла ещё не существовать
            pass
        store = _new_store()

        chunks = _load_chunks()
        if not chunks:
            raise ValueError(f"Файл базы знаний пуст: {KNOWLEDGE_FILE}")
        ids = [f"{COLLECTION_NAME}-{i:05d}" for i in range(len(chunks))]
        store.add_documents(chunks, ids=ids)
        _write_manifest(fingerprint, len(chunks))
        print("Индекс базы знаний пересобран.")
        return store

    print("Загружен актуальный индекс базы знаний.")
    return store


def search(query: str, vectorstore: Chroma | None, k: int = RERANKED_DOCUMENTS):
    """Возвращает список (document, rerank_score), отсортированный по релевантности."""
    if vectorstore is None or not query.strip():
        return []

    candidates = vectorstore.similarity_search(
        query, k=max(RETRIEVAL_CANDIDATES, k)
    )
    if not candidates:
        return []

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = get_reranker().predict(pairs, show_progress_bar=False)
    ranked = sorted(
        ((doc, float(score)) for doc, score in zip(candidates, scores)),
        key=lambda item: item[1],
        reverse=True,
    )

    if config.MIN_RERANK_SCORE is not None:
        filtered = [item for item in ranked if item[1] >= config.MIN_RERANK_SCORE]
        if filtered:
            ranked = filtered

    return ranked[:k]


def retrieve_and_rerank(query: str, vectorstore: Chroma | None):
    """Совместимость со старым API: только документы."""
    return [doc for doc, _ in search(query, vectorstore)]


def format_context(documents) -> str:
    if not documents:
        return "Подходящих фрагментов в локальной базе не найдено."

    formatted = []
    for index, doc in enumerate(documents, start=1):
        source = Path(doc.metadata.get("source", KNOWLEDGE_FILE.name)).name
        start = doc.metadata.get("start_index")
        location = f", символ {start}" if start is not None else ""
        formatted.append(
            f"[Фрагмент {index}; источник: {source}{location}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def status(vectorstore: Chroma | None = None) -> dict:
    """Краткое состояние индекса для /status и CLI."""
    manifest = _read_manifest()
    chunks = 0
    if vectorstore is not None:
        try:
            chunks = vectorstore._collection.count()
        except Exception:
            chunks = int(manifest.get("chunks", 0))
    fingerprint = manifest.get("fingerprint", {})
    return {
        "file": KNOWLEDGE_FILE.name,
        "file_size": KNOWLEDGE_FILE.stat().st_size if KNOWLEDGE_FILE.is_file() else 0,
        "chunks": chunks,
        "embedding_model": EMBEDDING_MODEL,
        "sha256": str(fingerprint.get("sha256", ""))[:12],
        "indexed_at": manifest.get("updated_at"),
        "collection": COLLECTION_NAME,
    }
