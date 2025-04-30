import argparse
import json
import os
from dotenv import load_dotenv

from db import init_db
from document_store import DocumentStore
from rag import RAG

# Load environment variables
load_dotenv()

def setup_database():
    """Set up the database with pgvector extension."""
    init_db()
    print("Database initialized with pgvector extension.")

def add_sample_documents():
    """Add sample documents to the document store."""
    document_store = DocumentStore()
    
    sample_docs = [
        {
            "content": "PostgreSQL is a powerful, open source object-relational database system with over 30 years of active development that has earned it a strong reputation for reliability, feature robustness, and performance.",
            "metadata": {"source": "postgresql.org", "category": "database"}
        },
        {
            "content": "pgvector is a PostgreSQL extension for vector similarity search. It allows you to store vectors in a new data type and query them by cosine distance, Euclidean distance, or inner product.",
            "metadata": {"source": "github.com/pgvector/pgvector", "category": "extension"}
        },
        {
            "content": "Retrieval Augmented Generation (RAG) is a technique used to augment Large Language Model (LLM) knowledge with additional, often private or real-time, data.",
            "metadata": {"source": "research paper", "category": "ai"}
        },
        {
            "content": "Embeddings are numerical representations of text that capture semantic meaning. They allow us to compare pieces of text based on their meaning rather than just keywords.",
            "metadata": {"source": "machine learning guide", "category": "ai"}
        },
        {
            "content": "Vector databases specialize in storing and querying vector embeddings efficiently, offering features like approximate nearest neighbor (ANN) search to find similar vectors quickly.",
            "metadata": {"source": "database guide", "category": "database"}
        }
    ]
    
    doc_ids = document_store.add_documents(sample_docs)
    print(f"Added {len(doc_ids)} sample documents with IDs: {doc_ids}")

def search_documents(query, top_k=5):
    """Search for documents similar to the query."""
    document_store = DocumentStore()
    results = document_store.similarity_search(query, top_k=top_k)
    
    print(f"\nSearch results for query: '{query}'")
    print("=" * 50)
    
    for i, doc in enumerate(results):
        print(f"Result {i+1} [Similarity: {doc['similarity']:.4f}]")
        print(f"Content: {doc['content']}")
        if doc['metadata']:
            print(f"Metadata: {doc['metadata']}")
        print("-" * 50)

def rag_query(question, top_k=3):
    """Perform RAG query with the question."""
    rag = RAG()
    result = rag.query(question, top_k=top_k)
    
    print(f"\nRAG response for question: '{question}'")
    print("=" * 50)
    print(f"Answer: {result['answer']}")
    
    if result['sources']:
        print("\nSources:")
        for i, source in enumerate(result['sources']):
            print(f"{i+1}. {source['content']}")

def main():
    parser = argparse.ArgumentParser(description="pgvector RAG Demo")
    parser.add_argument("--setup", action="store_true", help="Set up database and add sample documents")
    parser.add_argument("--search", help="Search for documents similar to query")
    parser.add_argument("--query", help="Perform RAG query with question")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_database()
        add_sample_documents()
    
    if args.search:
        search_documents(args.search, args.top_k)
    
    if args.query:
        rag_query(args.query, args.top_k)
    
    # If no arguments provided, show help
    if not (args.setup or args.search or args.query):
        parser.print_help()

if __name__ == "__main__":
    main() 