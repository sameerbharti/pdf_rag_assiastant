from app.retrieval.vector_store import VectorStore


INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/index/metadata.json"


def load_vector_store() -> VectorStore:
    """
    Load the persisted FAISS vector store.
    """

    return VectorStore.load(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )