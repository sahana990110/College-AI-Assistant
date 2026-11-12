"""Create a FAISS index from the college PDFs in the data directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfReadError

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class IngestionReport:
    """Summary of a completed knowledge-base rebuild."""

    document_count: int
    chunk_count: int
    failures: tuple[str, ...]


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create the shared, CPU-friendly sentence-transformer embedder."""
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True},
    )


def load_pdf_documents(data_dir: Path = DATA_DIR) -> tuple[list[Document], list[str]]:
    """Extract pages from PDFs, reporting unreadable files without hiding errors."""
    pdf_paths = sorted(path for path in data_dir.glob("*.pdf") if path.is_file())
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF documents were found in {data_dir}. Add college PDFs and try again."
        )

    documents: list[Document] = []
    failures: list[str] = []
    for pdf_path in pdf_paths:
        try:
            pages = PyPDFLoader(pdf_path).load()
            for page in pages:
                page.metadata["source"] = pdf_path.name
            documents.extend(pages)
        except (OSError, PdfReadError, ValueError) as exc:
            failures.append(f"Could not read {pdf_path.name}: {exc}")

    return documents, failures


def rebuild_vectorstore(
    data_dir: Path = DATA_DIR,
    vectorstore_dir: Path = VECTORSTORE_DIR,
) -> IngestionReport:
    """Rebuild and save the complete index, including all readable PDFs."""
    pages, failures = load_pdf_documents(data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        details = " ".join(failures)
        raise RuntimeError(
            "No readable text was extracted from the PDFs. "
            f"Check whether they contain selectable text. {details}".strip()
        )

    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(
        chunks,
        create_embeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    vectorstore.save_local(str(vectorstore_dir))

    indexed_documents = {
        chunk.metadata.get("source", "") for chunk in chunks
    } - {""}
    return IngestionReport(
        document_count=len(indexed_documents),
        chunk_count=len(chunks),
        failures=tuple(failures),
    )


if __name__ == "__main__":
    try:
        report = rebuild_vectorstore()
        print(
            f"Indexed {report.chunk_count} chunks from "
            f"{report.document_count} PDF(s) in {VECTORSTORE_DIR}."
        )
        for failure in report.failures:
            print(f"Warning: {failure}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Ingestion failed: {exc}") from exc
