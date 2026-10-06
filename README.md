# RAG-Based AI Document Assistant

A Retrieval-Augmented Generation (RAG) document assistant that allows users to upload PDF documents and ask questions about their contents.

The system retrieves relevant document chunks using semantic vector search and generates answers using Google Gemini. If Gemini generation is temporarily unavailable, the system uses a deterministic extractive fallback based on the retrieved document content.

---

## Features

- Upload PDF documents through a Streamlit interface
- Extract text from PDF files
- Split extracted text into chunks
- Generate semantic embeddings using `all-MiniLM-L6-v2`
- Store embeddings in a persistent FAISS index
- Retrieve relevant document chunks using semantic similarity
- Apply a relevance threshold before answering
- Generate document-grounded answers using Gemini
- Deterministic extractive fallback when Gemini generation fails
- Detect duplicate PDF documents using SHA-256 content hashes
- Persist the FAISS index and document metadata
- Display retrieved source documents, pages, chunks, and relevance scores
- Support responsibility-specific deterministic extraction

---

## Architecture

```text
                         PDF Document
                              |
                              v
                       PDF Text Loader
                              |
                              v
                        Text Chunker
                              |
                              v
                    MiniLM Embedding Model
                              |
                              v
                       FAISS Vector Store
                              |
                              v
                     Semantic Retrieval
                              |
                              v
                       Relevance Gate
                              |
                              v
                       Context Builder
                              |
                              v
                    Gemini Generation
                              |
                    +---------+---------+
                    |                   |
                    | Gemini succeeds  | Gemini fails
                    |                   |
                    v                   v
                  Answer       Extractive Fallback
                    |                   |
                    +---------+---------+
                              |
                              v
                       Answer + Sources
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.12 |
| User Interface | Streamlit |
| PDF Processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Search | FAISS CPU |
| Generation | Google Gemini |
| Environment Variables | python-dotenv |
| Numerical Processing | NumPy |

---

## Project Structure

```text
rag-document-assistant/
│
├── app/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── config.py
│   │
│   ├── generation/
│   │   ├── generator.py
│   │   ├── prompt_builder.py
│   │   └── extractive_fallback.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   │
│   └── retrieval/
│       ├── embeddings.py
│       ├── index_loader.py
│       ├── retriever.py
│       └── vector_store.py
│
├── data/
│   ├── documents/
│   └── index/
│       ├── faiss.index
│       └── metadata.json
│
├── tests/
│   └── build_index.py
│
├── test_embeddings.py
├── test_fallback.py
├── test_generation.py
├── test_index_loader.py
├── test_loader.py
├── test_local_generation.py
├── test_local_model.py
├── test_persisted_retrieval.py
├── test_persistence.py
├── test_prompt_builder.py
├── test_rag_pipeline.py
├── test_retriever.py
├── test_vector_store.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requirements

The project currently uses the following pinned dependencies:

```text
pymupdf==1.28.2
sentence-transformers==6.1.0
faiss-cpu==1.15.1
numpy==2.5.3
python-dotenv==1.2.4
streamlit==1.65.0
google-genai==2.28.0
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

The application uses a Gemini API key for generation.

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The project also contains:

```text
.env.example
```

with the following template:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Do not commit your actual API key to Git.

---

## Running the Application

From the project root, start the Streamlit application with:

```bash
python -m streamlit run app/main.py
```

The application opens the **RAG-Based AI Document Assistant** interface.

---

## Using the Application

### 1. Upload a PDF

Use the **Upload Document** section to select a PDF file.

After selecting the document, click:

```text
Add Document to Knowledge Base
```

The application then:

1. Loads the existing FAISS vector store.
2. Checks whether the PDF content has already been indexed.
3. Calculates a SHA-256 hash for duplicate detection.
4. Extracts text from the PDF.
5. Creates text chunks.
6. Generates embeddings for the chunks.
7. Adds the new embeddings to the existing FAISS index.
8. Saves the updated vector index and metadata.

If the same PDF content has already been indexed, it is not added again.

---

### 2. Ask a Question

Enter a question in the **Ask a question** field.

For example:

```text
How do I verify my student status?
```

The application retrieves relevant document chunks and generates an answer using the retrieved context.

The interface displays:

- Answer
- Source document
- Page number
- Chunk ID
- Relevance score
- Retrieved text

---

## RAG Pipeline

The main RAG pipeline is implemented in:

```text
app/rag_pipeline.py
```

The pipeline follows this sequence:

```text
User Question
      |
      v
Question Embedding
      |
      v
FAISS Retrieval
      |
      v
Relevance Check
      |
      v
Relevant Chunks
      |
      v
Context Builder
      |
      v
Gemini Generator
      |
      v
Answer
```

The pipeline returns both the generated answer and the retrieved source information.

---

## Retrieval Configuration

The current RAG pipeline uses:

```python
top_k = 2
```

and:

```python
MIN_RELEVANCE_SCORE = 0.20
```

The relevance threshold prevents the generation stage from answering questions using chunks that are not sufficiently relevant.

If no sufficiently relevant information is retrieved, the system returns:

```text
I could not find enough information in the provided documents to answer this question.
```

---

## Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

through Sentence Transformers.

The embedding model converts both document chunks and user queries into numerical vectors.

These vectors are used by FAISS for semantic similarity search.

---

## FAISS Vector Store

FAISS is used for local vector similarity search.

The persisted vector index is stored at:

```text
data/index/faiss.index
```

Document metadata is stored at:

```text
data/index/metadata.json
```

The persisted index allows the application to reuse the existing knowledge base when the application is restarted.

---

## PDF Ingestion

PDF ingestion is implemented using:

```text
app/ingestion/loader.py
```

The extracted text is passed to:

```text
app/ingestion/chunker.py
```

which creates the chunks used for embedding and retrieval.

The indexing workflow is implemented in:

```text
app/ingestion/indexer.py
```

---

## Duplicate PDF Detection

The indexer uses SHA-256 hashing to detect duplicate PDF content.

Before adding a document to the index, the system:

1. Calculates the SHA-256 hash of the uploaded PDF.
2. Checks the already indexed document sources.
3. Calculates their hashes.
4. Compares the hashes.
5. Skips indexing if identical content already exists.

This prevents the same document from being indexed multiple times.

---

## Generation

Gemini generation is implemented in:

```text
app/generation/generator.py
```

The current model configuration is:

```text
gemini-3.8-flash
```

The generator receives:

- User question
- Retrieved document context
- Retrieved result metadata

The generation prompt instructs Gemini to:

- Answer using only the provided document context.
- Avoid outside knowledge.
- Avoid unsupported assumptions.
- Avoid inventing information.
- Preserve relevant procedural steps.
- Preserve the original order of procedural steps.
- Preserve important document-specific details.
- Answer only the requested information category.
- Keep answers concise and focused.
- State when the provided context is insufficient.

---

## Deterministic Fallback

The application does not depend entirely on Gemini being available.

If Gemini generation raises an exception, the generator uses:

```text
app/generation/extractive_fallback.py
```

to extract relevant sentences from the retrieved document chunks.

This provides a document-grounded answer even when Gemini temporarily fails.

For example, if Gemini returns a temporary service error, the RAG pipeline can still return relevant information from the retrieved document.

---

## Responsibility Extraction

Responsibility-related questions have a dedicated deterministic extraction path.

When a question contains a responsibility-related query, the generator attempts to use:

```text
extract_responsibilities()
```

before sending the request to Gemini.

This is designed to preserve responsibility items from the document and return them as separate bullet points.

---

## Context Building

Retrieved chunks are converted into the generation context by:

```text
app/generation/prompt_builder.py
```

Only sufficiently relevant retrieved chunks are passed to the context-building stage.

This helps keep the generation context focused on the user's question.

---

## Out-of-Context Questions

The RAG pipeline contains a relevance gate.

If the best retrieved result has a score below:

```text
0.20
```

the system does not attempt normal generation.

Instead, it returns:

```text
I could not find enough information in the provided documents to answer this question.
```

This helps prevent answers based on unrelated document content.

---

## Testing

The project contains tests covering the major components of the RAG system.

Available test scripts include:

```text
test_embeddings.py
test_fallback.py
test_generation.py
test_index_loader.py
test_loader.py
test_local_generation.py
test_local_model.py
test_persisted_retrieval.py
test_persistence.py
test_prompt_builder.py
test_rag_pipeline.py
test_retriever.py
test_vector_store.py
```

There is also an index-building utility:

```text
tests/build_index.py
```

---

## Generation Integration Test

The end-to-end generation test can be run with:

```bash
python test_generation.py
```

The test performs:

```text
PDF Loading
    ↓
Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
Context Building
    ↓
Generator
    ↓
Answer
    ↓
Retrieved Sources
```

The test also exercises the generation fallback when Gemini is unavailable.

---

## Current Project Status

The following major components have been tested:

- PDF loading
- Text chunking
- Embedding generation
- FAISS retrieval
- Vector-store persistence
- Persisted retrieval
- Prompt/context building
- Relevance gating
- Extractive fallback
- Gemini generation
- Responsibility extraction
- Full RAG pipeline
- PDF upload and indexing
- SHA-256 duplicate document detection
- Source display

---

## Data Directories

### Documents

Uploaded PDFs are stored in:

```text
data/documents/
```

### Vector Index

The persistent FAISS index is stored in:

```text
data/index/faiss.index
```

### Metadata

The document metadata associated with the vector index is stored in:

```text
data/index/metadata.json
```

---

## Important Notes

### Gemini API

Gemini generation requires a valid:

```text
GEMINI_API_KEY
```

in the `.env` file.

If Gemini temporarily becomes unavailable, the application can use the deterministic extractive fallback.

### Embedding Model

The first use of the embedding model may require downloading:

```text
all-MiniLM-L6-v2
```

from Hugging Face.

### Persistent Index

The FAISS index and metadata should be kept together:

```text
data/index/faiss.index
data/index/metadata.json
```

They represent the persisted knowledge base.

---

## Git and Secrets

The real `.env` file should not be committed to the repository.

The repository should use:

```text
.env.example
```

as the configuration template.

Never place a real Gemini API key inside:

- `README.md`
- Python source files
- Test files
- Git commits
- Public repositories

---

## License

This project is intended for educational and project-development purposes.