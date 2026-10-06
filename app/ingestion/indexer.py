from pathlib import Path
import hashlib

import numpy as np

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore


INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/index/metadata.json"


def _file_sha256(file_path: str) -> str:
    """
    Calculate the SHA-256 hash of a file.
    """

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


def _document_already_indexed(
    pdf_path: str,
    vector_store: VectorStore
) -> bool:
    """
    Check whether the same PDF content already exists
    in the indexed documents.
    """

    uploaded_hash = _file_sha256(
        pdf_path
    )

    checked_sources = set()

    for document in vector_store.documents:

        source = document["source"]

        if source in checked_sources:
            continue

        checked_sources.add(source)

        source_path = Path(source)

        if not source_path.exists():
            continue

        try:
            existing_hash = _file_sha256(
                str(source_path)
            )
        except OSError:
            continue

        if existing_hash == uploaded_hash:
            return True

    return False


def add_pdf_to_index(
    pdf_path: str,
    embedding_model: EmbeddingModel | None = None
) -> int:
    """
    Load a PDF, create chunks, generate embeddings,
    and append the document to the existing FAISS index.

    Identical PDF files are detected using SHA-256
    content hashes and are not indexed twice.

    Returns:
        Number of newly added chunks.
    """

    pdf_path = str(
        Path(pdf_path)
    )

    print(
        f"Indexing document: {pdf_path}"
    )

    # --------------------------------------------------
    # 1. Load existing vector store
    # --------------------------------------------------

    vector_store = VectorStore.load(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    # --------------------------------------------------
    # 2. Check for identical PDF content
    # --------------------------------------------------

    if _document_already_indexed(
        pdf_path,
        vector_store
    ):
        print(
            "Document content already exists "
            "in the index."
        )

        print(
            f"Total vectors: "
            f"{vector_store.index.ntotal}"
        )

        return 0

    # --------------------------------------------------
    # 3. Load PDF
    # --------------------------------------------------

    pages = load_pdf(
        pdf_path
    )

    if not pages:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    # --------------------------------------------------
    # 4. Create chunks
    # --------------------------------------------------

    chunks = chunk_text(
        pages
    )

    if not chunks:
        raise ValueError(
            "No chunks were created from the PDF."
        )

    # --------------------------------------------------
    # 5. Load embedding model
    # --------------------------------------------------

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    # --------------------------------------------------
    # 6. Generate embeddings
    # --------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # --------------------------------------------------
    # 7. Append new chunks
    # --------------------------------------------------

    vector_store.add(
        embeddings=embeddings,
        documents=chunks
    )

    # --------------------------------------------------
    # 8. Save updated index
    # --------------------------------------------------

    vector_store.save(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    print(
        f"Added {len(chunks)} new chunks."
    )

    print(
        f"Total vectors: "
        f"{vector_store.index.ntotal}"
    )

    return len(chunks)