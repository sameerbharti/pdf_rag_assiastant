from app.rag_pipeline import RAGPipeline


print("=" * 60)
print("COMPLETE RAG PIPELINE TEST")
print("=" * 60)


# --------------------------------------------------
# 1. Initialize pipeline
# --------------------------------------------------

pipeline = RAGPipeline(
    top_k=3
)


# --------------------------------------------------
# 2. Ask question
# --------------------------------------------------

query = "How do I verify my student status?"

result = pipeline.ask(
    query
)


# --------------------------------------------------
# 3. Display answer
# --------------------------------------------------

print("\n" + "=" * 60)
print("QUESTION")
print("=" * 60)

print(result["query"])


print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(result["answer"])


# --------------------------------------------------
# 4. Display sources
# --------------------------------------------------

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for i, source in enumerate(
    result["sources"],
    start=1
):

    filename = (
        source["source"]
        .replace("\\", "/")
        .split("/")[-1]
    )

    print(
        f"{i}. {filename} "
        f"| Page {source['page']} "
        f"| Chunk {source['chunk_id']} "
        f"| Score {source['score']:.4f}"
    )