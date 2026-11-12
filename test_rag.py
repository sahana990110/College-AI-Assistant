"""Offline tests for indexing, retrieval, fallback, and source reporting."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

import ingest
import rag


class KeywordEmbeddings(Embeddings):
    """Small deterministic embedder for tests; it does not download a model."""

    terms = ("computer", "fees", "hostel", "exams")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        text = text.lower()
        values = [float(term in text) for term in self.terms]
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values] if norm else [0.0] * len(values)


class CollegeRAGTests(unittest.TestCase):
    def test_ingest_retrieve_and_report_source_without_groq(self) -> None:
        embeddings = KeywordEmbeddings()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            index_dir = root / "vectorstore"
            data_dir.mkdir()
            (data_dir / "courses.pdf").write_bytes(b"fixture")

            page = Document(
                page_content=(
                    "Computer science eligibility requires mathematics "
                    "in grade 12."
                ),
                metadata={"page": 0},
            )
            with (
                patch("ingest.PyPDFLoader") as pdf_loader,
                patch("ingest.create_embeddings", return_value=embeddings),
            ):
                pdf_loader.return_value.load.return_value = [page]
                report = ingest.rebuild_vectorstore(data_dir, index_dir)

            self.assertEqual(report.document_count, 1)
            self.assertGreater(report.chunk_count, 0)

            captured_context: list[str] = []

            def answer_from_context(prompt_value: PromptValue) -> AIMessage:
                messages = prompt_value.to_messages()
                captured_context.append(str(messages[0].content))
                return AIMessage(content="Mathematics in grade 12 is required.")

            fake_model = RunnableLambda(answer_from_context)
            with (
                patch("rag.create_embeddings", return_value=embeddings),
                patch("rag.ChatGroq", return_value=fake_model),
                patch.dict(
                    os.environ,
                    {
                        "GROQ_API_KEY": "test-key-not-sent-to-a-network",
                        "GROQ_MODEL": "test-model",
                    },
                ),
            ):
                service = rag.CollegeRAG(index_dir)
                result = service.answer("What is the computer science eligibility?")
                missing = service.answer("What are the library opening hours?")

            self.assertEqual(result.answer, "Mathematics in grade 12 is required.")
            self.assertEqual(result.sources, ["courses.pdf"])
            self.assertIn("mathematics in grade 12", captured_context[0].lower())
            self.assertEqual(missing.answer, rag.FALLBACK_ANSWER)
            self.assertEqual(missing.sources, [])


if __name__ == "__main__":
    unittest.main()
