import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# PDF LOADING
# ============================================================

from unstructured.partition.pdf import partition_pdf

# ============================================================
# CHUNKING
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# OPENAI EMBEDDINGS
# ============================================================

from langchain_openai import OpenAIEmbeddings

# ============================================================
# PINECONE
# ============================================================

from pinecone import Pinecone, ServerlessSpec


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ergo-claims")

PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ergo-claims-dev")

DOCUMENTS_DIR = Path("documents")


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536



# To one of these valid Pinecone models:
RERANK_MODEL = "bge-reranker-v2-m3"  # or "pinecone-rerank-v0"


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file.")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing. Please add it to your .env file."
    )


# ============================================================
# CLIENT INITIALIZATION
# ============================================================

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)


# ============================================================
# LOAD PDF DOCUMENTS
# ============================================================

def load_documents():
    """Load PDF documents from the documents folder using Unstructured."""
    documents = []

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"Documents folder not found: {DOCUMENTS_DIR}")

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found inside documents folder.")

    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file.name}")
        try:
            elements = partition_pdf(filename=str(pdf_file), strategy="auto")

            for element in elements:
                text = str(element).strip()
                if text:
                    documents.append({"text": text, "source": pdf_file.name})

        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")

    print(f"Loaded {len(documents)} document elements.")
    return documents


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(documents):
    """Create chunks from extracted PDF text."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for document in documents:
        text = document["text"]
        source = document["source"]

        split_texts = splitter.split_text(text)

        for index, chunk in enumerate(split_texts):
            chunk = chunk.strip()
            if not chunk:
                continue

            chunks.append(
                {"text": chunk, "source": source, "chunk_id": index}
            )

    print(f"Created {len(chunks)} chunks.")
    return chunks


# ============================================================
# CREATE OPENAI DENSE EMBEDDINGS
# ============================================================

def create_dense_embeddings(texts):
    """Create OpenAI embeddings for document chunks."""
    print("Creating OpenAI dense embeddings...")

    if not texts:
        raise ValueError("No text provided for embedding.")

    all_vectors = []
    batch_size = 100
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        end = min(start + batch_size, total)

        print(f"Embedding batch {start + 1}-{end}/{total}")
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)

    print(f"Created {len(all_vectors)} embeddings.")
    return all_vectors


# ============================================================
# CONNECT TO PINECONE
# ============================================================

def get_pinecone_index():
    print(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")

    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(PINECONE_INDEX_NAME)


# ============================================================
# UPSERT DOCUMENTS TO PINECONE
# ============================================================

def upsert_to_pinecone(chunks, vectors, index):
    print("Uploading vectors to Pinecone...")

    records = []

    for chunk, vector in zip(chunks, vectors):
        vector_id = (
            f"{chunk['source']}-{chunk['chunk_id']}-{uuid.uuid4().hex[:8]}"
        )

        records.append(
            {
                "id": vector_id,
                "values": vector,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                },
            }
        )

    batch_size = 100
    total = len(records)

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
        end = min(start + batch_size, total)
        print(f"Uploaded {end}/{total}")

    print("Pinecone upload completed.")


# ============================================================
# INGESTION PIPELINE
# ============================================================

def ingest_documents():
    print("\n" + "=" * 60)
    print("ERGO CLAIM RAG INGESTION")
    print("=" * 60)

    documents = load_documents()
    chunks = create_chunks(documents)
    texts = [chunk["text"] for chunk in chunks]
    vectors = create_dense_embeddings(texts)
    index = get_pinecone_index()

    upsert_to_pinecone(chunks, vectors, index)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED")
    print("=" * 60)
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Pinecone index: {PINECONE_INDEX_NAME}")
    print(f"Namespace: {PINECONE_NAMESPACE}")


# ============================================================
# QUERY EMBEDDING & PINECONE RERANKING
# ============================================================

def create_query_embedding(query):
    """Convert the user's query into an OpenAI embedding."""
    return embeddings.embed_query(query)


def rerank_results(query: str, initial_matches: list, top_n: int):
    """Rerank candidate matches using Pinecone's Inference API."""
    if not initial_matches:
        return []

    # Prepare document objects for Pinecone Inference Rerank
    documents = [
        {
            "id": match["id"],
            "text": match["metadata"]["text"],
            "source": match["metadata"].get("source", ""),
            "chunk_id": match["metadata"].get("chunk_id", None),
        }
        for match in initial_matches
        if "metadata" in match and "text" in match["metadata"]
    ]

    # Perform reranking
    response = pc.inference.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=True,
    )

    reranked_results = []
    for hit in response.data:
        reranked_results.append(
            {
                "id": hit.document["id"],
                "score": hit.score,
                "text": hit.document["text"],
                "source": hit.document.get("source", ""),
                "chunk_id": hit.document.get("chunk_id", None),
            }
        )

    return reranked_results


# ============================================================
# RAG SEARCH (WITH RERANKING)
# ============================================================

def rag_search(
    query: str, top_k: int = 25, top_n: int = 5, use_reranker: bool = True
):
    """Search ERGO insurance claim documents with optional Pinecone reranking.

    :param query: Search string.
    :param top_k: Number of candidate documents retrieved via dense vector
        search.
    :param top_n: Final number of reranked documents returned.
    :param use_reranker: Flag to toggle Pinecone Reranking.
    """
    if not query:
        return {"success": False, "error": "Query cannot be empty."}

    try:
        index = get_pinecone_index()
        query_vector = create_query_embedding(query)

        # 1. Semantic Vector Search
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
        )

        matches = results.get("matches", [])

        # 2. Reranking Step
        if use_reranker and matches:
            final_results = rerank_results(
                query=query, initial_matches=matches, top_n=top_n
            )
        else:
            # Fallback to standard vector similarity scores
            final_results = [
                {
                    "id": m.get("id"),
                    "score": m.get("score", 0),
                    "text": m.get("metadata", {}).get("text", ""),
                    "source": m.get("metadata", {}).get("source", ""),
                    "chunk_id": m.get("metadata", {}).get("chunk_id"),
                }
                for m in matches[:top_n]
            ]

        return {"success": True, "query": query, "results": final_results}

    except Exception as e:
        return {"success": False, "query": query, "error": str(e)}


# ============================================================
# TEST SEARCH
# ============================================================

def test_search():
    print("\n" + "=" * 60)
    print("TESTING RAG SEARCH WITH PINECONE RERANKER")
    print("=" * 60)

    query = "What is the coverage limit for a motor insurance claim?"

    result = rag_search(query=query, top_k=20, top_n=5, use_reranker=True)

    print("\nQuery:", query)
    print("\nResults:")

    if not result["success"]:
        print("ERROR:", result["error"])
        return

    for i, item in enumerate(result["results"], start=1):
        print("\n" + "-" * 50)
        print(f"Result {i}")
        print(f"Rerank Score: {item['score']:.4f}")
        print(f"Source: {item['source']}")
        print(f"Chunk: {item['chunk_id']}")
        print(f"Text: {item['text'][:300]}...")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Uncomment to ingest documents:
    # ingest_documents()

    test_search()