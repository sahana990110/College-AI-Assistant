"""Streamlit interface for the College AI Assistant."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ingest import rebuild_vectorstore
from rag import CollegeRAG

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def get_rag_service() -> CollegeRAG:
    """Load embeddings, the local index, and the configured Groq model once."""
    return CollegeRAG()


def available_documents() -> list[str]:
    """Return the PDF filenames currently present in the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.name for path in DATA_DIR.glob("*.pdf") if path.is_file())


def save_uploaded_pdfs(uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> list[str]:
    """Save uploaded PDFs using only their filenames, never their supplied paths."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_names: list[str] = []

    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        if not filename.lower().endswith(".pdf"):
            raise ValueError(f"{filename} is not a PDF file.")
        if not uploaded_file.getvalue().startswith(b"%PDF"):
            raise ValueError(f"{filename} does not have a valid PDF header.")

        destination = DATA_DIR / filename
        destination.write_bytes(uploaded_file.getvalue())
        saved_names.append(filename)

    return saved_names


st.markdown(
    """
    <style>
    .block-container { max-width: 1050px; padding-top: 2rem; }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(120, 130, 150, 0.18);
        border-radius: 14px;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🎓 College AI Assistant")
    st.caption("A document-grounded assistant for college information.")
    st.markdown(
        "Answers are generated from the college PDFs in `data/`. "
        "If the documents do not provide an answer, the assistant will say so."
    )
    st.divider()
    st.subheader("Knowledge base")
    document_names = available_documents()
    if document_names:
        for document_name in document_names:
            st.markdown(f"- `{document_name}`")
    else:
        st.info("No PDFs yet. Upload documents below or add them to `data/`.")

    st.subheader("Add or update PDFs")
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Select college documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Uploaded files are saved in data/ and included in a rebuilt index.",
        )
        upload_submitted = st.form_submit_button(
            "Upload and update knowledge base",
            disabled=not uploaded_files,
            use_container_width=True,
        )

    if upload_submitted and uploaded_files:
        try:
            saved_names = save_uploaded_pdfs(uploaded_files)
            report = rebuild_vectorstore()
            get_rag_service.clear()
            if report.document_count == 0:
                st.error("No readable PDF text was found. Check the uploaded documents.")
            else:
                st.success(
                    f"Indexed {report.chunk_count} text chunks from "
                    f"{report.document_count} PDF(s)."
                )
                st.caption(f"Saved: {', '.join(saved_names)}")
            for failure in report.failures:
                st.warning(failure)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Could not update the knowledge base: {exc}")

    st.divider()
    st.caption(
        "Built with Streamlit, LangChain, Hugging Face Sentence Transformers, "
        "FAISS, PyPDF, and Groq."
    )

st.title("College AI Assistant")
st.markdown(
    "Ask questions about courses, exams, fees, departments, faculty, rules, "
    "hostel, notices, and other college information."
)

with st.expander("Example questions"):
    st.markdown(
        """
        - What departments are available in the college?
        - What are the eligibility requirements for CSE?
        - What is the examination procedure?
        - What are the hostel rules?
        - What courses are offered?
        - What documents are required for admission?
        - What is the fee structure?
        - Who is the HOD of the CSE department?
        """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

clear_column, _ = st.columns([1, 5])
with clear_column:
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Source documents"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

question = st.chat_input("Ask a question about your college...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching college documents..."):
                result = get_rag_service().answer(question)
            st.markdown(result.answer)
            if result.sources:
                with st.expander("Source documents"):
                    for source in result.sources:
                        st.markdown(f"- `{source}`")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                }
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            error_message = str(exc)
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "sources": []}
            )
        except Exception as exc:
            error_message = (
                "The request could not be completed. Check your Groq API key, "
                f"configured model, network connection, and application logs. ({exc})"
            )
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "sources": []}
            )
