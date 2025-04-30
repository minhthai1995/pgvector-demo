import json
from sqlalchemy import text
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from db import Document, get_db
from embeddings import Embedder
from config import DATABASE_URL

class DocumentStore:
    def __init__(self):
        """Initialize document store with embedder."""
        self.embedder = Embedder()
    
    def add_document(self, content, metadata=None):
        """Add a document to the store with its embedding."""
        embedding = self.embedder.embed_text(content)
        
        # Convert metadata to JSON string if provided
        metadata_str = None
        if metadata:
            metadata_str = json.dumps(metadata)
        
        # Create new document
        db = get_db()
        doc = Document(
            content=content,
            doc_metadata=metadata_str,
            embedding=embedding
        )
        
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        return doc.id
    
    def add_documents(self, documents):
        """Add multiple documents to the store with their embeddings."""
        if not documents:
            return []
        
        # Process documents
        processed_docs = []
        for doc in documents:
            if isinstance(doc, str):
                processed_docs.append({"content": doc, "metadata": None})
            elif isinstance(doc, dict) and "content" in doc:
                metadata = doc.get("metadata")
                processed_docs.append({"content": doc["content"], "metadata": metadata})
        
        # Get embeddings for all documents
        contents = [doc["content"] for doc in processed_docs]
        embeddings = self.embedder.embed_documents(contents)
        
        # Add all documents to database
        db = get_db()
        doc_ids = []
        
        for doc, embedding in zip(processed_docs, embeddings):
            # Convert metadata to JSON string if provided
            metadata_str = None
            if doc["metadata"]:
                metadata_str = json.dumps(doc["metadata"])
            
            db_doc = Document(
                content=doc["content"],
                doc_metadata=metadata_str,
                embedding=embedding
            )
            
            db.add(db_doc)
            db.flush()  # Flush to get ID without committing transaction
            doc_ids.append(db_doc.id)
        
        db.commit()
        return doc_ids
    
    def similarity_search(self, query, top_k=5):
        """Search for documents similar to the query."""
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Connect directly with psycopg2 for pgvector operations
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Convert numpy array to list
        query_vector = query_embedding.tolist()
        
        # Format as PostgreSQL array - use brackets for pgvector
        vector_str = "[" + ",".join([str(x) for x in query_vector]) + "]"
        
        # Execute query
        cursor.execute(f"""
            SELECT id, content, doc_metadata, 
                   1 - (embedding <=> '{vector_str}'::vector) as similarity
            FROM documents
            ORDER BY embedding <=> '{vector_str}'::vector
            LIMIT {top_k}
        """)
        
        results = cursor.fetchall()
        
        # Process results
        documents = []
        for row in results:
            metadata = json.loads(row[2]) if row[2] else None
            documents.append({
                "id": row[0],
                "content": row[1],
                "metadata": metadata,
                "similarity": float(row[3])
            })
        
        # Close connections
        cursor.close()
        conn.close()
        
        return documents
    
    def get_document(self, doc_id):
        """Retrieve a document by its ID."""
        db = get_db()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        
        if not doc:
            return None
        
        metadata = json.loads(doc.doc_metadata) if doc.doc_metadata else None
        
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata
        } 