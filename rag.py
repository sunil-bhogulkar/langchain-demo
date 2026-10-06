"""In-memory vector store with Ollama embeddings."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "nomic-embed-text"
DATA_DIR = Path(__file__).parent / "data"

_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
_embeddings = OllamaEmbeddings(model=EMBED_MODEL)
store = InMemoryVectorStore(embedding=_embeddings)


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    docs = []
    for path in sorted(data_dir.glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        docs.append(
            Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        )
    return docs


def ingest(data_dir: Path = DATA_DIR) -> int:
    """Chunk and embed all documents into the in-memory store."""
    docs = load_documents(data_dir)
    if not docs:
        return 0
    chunks = _splitter.split_documents(docs)
    store.add_documents(chunks)
    return len(chunks)


def retrieve(query: str, k: int = 3, min_score: float = 0.5) -> list[Document]:
    docs = store.similarity_search_with_score(query, k=k)
    return [doc for doc, score in docs if score >= min_score]
