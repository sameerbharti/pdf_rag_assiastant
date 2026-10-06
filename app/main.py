import streamlit as st
from pathlib import Path

from app.rag_pipeline import RAGPipeline
from app.ingestion.indexer import add_pdf_to_index


st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)


st.title("RAG-Based AI Document Assistant")

st.write(
    "Upload a PDF and ask questions about its contents."
)


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


pipeline = load_pipeline()


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)


if uploaded_file is not None:

    documents_dir = Path(
        "data/documents"
    )

    documents_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    uploaded_path = (
        documents_dir / uploaded_file.name
    )

    with open(
        uploaded_path,
        "wb"
    ) as file:
        file.write(
            uploaded_file.getbuffer()
        )

    if st.button("Add Document to Knowledge Base"):

        with st.spinner(
            "Processing document..."
        ):

            chunks_added = add_pdf_to_index(
                str(uploaded_path),
                embedding_model=pipeline.embedding_model
            )

        st.success(
            f"Document added successfully. "
            f"{chunks_added} chunks indexed."
        )

        st.cache_resource.clear()

        st.rerun()


# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

query = st.text_input(
    "Ask a question",
    placeholder="How do I verify my student status?"
)


if query:

    result = pipeline.ask(
        query
    )

    st.subheader("Answer")

    st.write(
        result["answer"]
    )

    st.subheader("Sources")

    for i, source in enumerate(
        result["sources"],
        start=1
    ):

        filename = (
            source["source"]
            .replace("\\", "/")
            .split("/")[-1]
        )

        with st.expander(
            f"Source {i} — {filename} "
            f"(Page {source['page']}, "
            f"Score {source['score']:.4f})"
        ):

            st.write(
                f"**Document:** {filename}"
            )

            st.write(
                f"**Page:** {source['page']}"
            )

            st.write(
                f"**Chunk:** {source['chunk_id']}"
            )

            st.write(
                f"**Relevance score:** "
                f"{source['score']:.4f}"
            )

            st.write(
                "**Retrieved text:**"
            )

            st.write(
                source["text"]
            )