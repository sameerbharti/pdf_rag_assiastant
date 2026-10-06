from app.retrieval.index_loader import load_vector_store
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import Retriever
from app.generation.prompt_builder import build_context
from app.generation.generator import Generator


class RAGPipeline:
    MIN_RELEVANCE_SCORE = 0.20

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

        print("Initializing RAG pipeline...")

        self.vector_store = load_vector_store()

        self.embedding_model = EmbeddingModel()

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model
        )

        self.generator = Generator()

        print("RAG pipeline ready.")

    def ask(self, query: str) -> dict:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        results = self.retriever.retrieve(
            query=query,
            top_k=self.top_k
        )

        if not results or results[0]["score"] < self.MIN_RELEVANCE_SCORE:
            return {
                "query": query,
                "answer": (
                    "I could not find enough information in the "
                    "provided documents to answer this question."
                ),
                "sources": results
            }

        results = [
            result
            for result in results
            if result["score"] >= self.MIN_RELEVANCE_SCORE
        ]

        context = build_context(results)

        answer = self.generator.generate(
            query=query,
            context=context,
            results=results
        )

        return {
            "query": query,
            "answer": answer,
            "sources": results
        }