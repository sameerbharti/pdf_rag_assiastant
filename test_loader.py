from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text


PDF_PATH = "data/documents/test.pdf"


pages = load_pdf(PDF_PATH)

chunks = chunk_text(pages)


print("=" * 60)
print("PDF + METADATA CHUNKING TEST")
print("=" * 60)

print(f"Number of pages: {len(pages)}")
print(f"Number of chunks: {len(chunks)}")


for chunk in chunks:

    print("\n" + "=" * 60)

    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page']}")
    print(f"Source: {chunk['source']}")
    print(f"Characters: {len(chunk['text'])}")

    print("\nText:")
    print(chunk["text"])