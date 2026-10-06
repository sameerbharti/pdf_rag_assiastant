import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.documents = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[dict]
    ):
        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents."
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} "
                f"does not match index dimension {self.dimension}."
            )

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> list[dict]:

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            document = self.documents[index]

            results.append(
                {
                    "score": float(score),
                    "text": document["text"],
                    "page": document["page"],
                    "source": document["source"],
                    "chunk_id": document["chunk_id"],
                }
            )

        return results

    def save(
        self,
        index_path: str,
        metadata_path: str
    ):
        """
        Save FAISS index and document metadata.
        """

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save FAISS vectors
        faiss.write_index(
            self.index,
            str(index_path)
        )

        # Save chunk metadata
        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.documents,
                file,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(
        cls,
        index_path: str,
        metadata_path: str
    ):
        """
        Load FAISS index and document metadata.
        """

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        # Load FAISS index
        index = faiss.read_index(
            str(index_path)
        )

        # Load metadata
        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            documents = json.load(file)

        vector_store = cls(
            dimension=index.d
        )

        vector_store.index = index
        vector_store.documents = documents

        return vector_store