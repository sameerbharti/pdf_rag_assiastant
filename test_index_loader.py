from app.retrieval.index_loader import load_vector_store


vector_store = load_vector_store()


print("=" * 60)
print("PERSISTED INDEX LOADER TEST")
print("=" * 60)

print(f"Dimension: {vector_store.dimension}")
print(f"Vectors: {vector_store.index.ntotal}")
print(f"Metadata records: {len(vector_store.documents)}")

print("\nFirst document metadata:")

first = vector_store.documents[0]

print(f"Chunk ID: {first['chunk_id']}")
print(f"Page: {first['page']}")
print(f"Source: {first['source']}")
print(f"Text preview: {first['text'][:200]}")