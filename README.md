# PGVector RAG System with Docker

A Retrieval Augmented Generation (RAG) system using PostgreSQL with pgvector extension for efficient document embedding storage and semantic search. This implementation uses Docker to simplify the setup process and avoid the complexities of compiling pgvector from source.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quick Setup](#quick-setup)
- [Docker Setup Explained](#docker-setup-explained)
- [Core Components](#core-components)
  - [Database Layer](#database-layer)
  - [Embedding Engine](#embedding-engine)
  - [Document Store](#document-store)
  - [RAG Implementation](#rag-implementation)
- [Usage Examples](#usage-examples)
  - [Initialize Database](#initialize-database)
  - [Search for Similar Documents](#search-for-similar-documents)
  - [Query with RAG](#query-with-rag)
- [Adding Custom Documents](#adding-custom-documents)
- [Troubleshooting](#troubleshooting)
- [Shutting Down](#shutting-down)

## Features

- **Docker-based PostgreSQL with pgvector**: Pre-configured container with PostgreSQL 15 and pgvector extension
- **Semantic Document Embedding**: Document embedding using sentence-transformers for meaning-based retrieval
- **Vector Similarity Search**: Efficient nearest-neighbor search with cosine similarity
- **Customizable RAG Pipeline**: Implementation using OpenAI for generating answers from retrieved context
- **Command-line Interface**: Simple CLI for database setup, search, and querying
- **Metadata Support**: Store and search with metadata associated with documents

## Prerequisites

- Docker and Docker Compose (for running PostgreSQL with pgvector)
- Python 3.8+
- OpenAI API key (for RAG functionality)

## Project Structure

Here's the overall project structure and what each file does:

```
PGVector/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Docker configuration for PostgreSQL with pgvector
├── .env                      # Environment variables (not tracked in git)
├── config.py                 # Configuration settings for database and models
├── db.py                     # Database connection and ORM models
├── embeddings.py             # Document embedding functionality
├── document_store.py         # Document storage and vector search
├── rag.py                    # Retrieval Augmented Generation implementation
└── main.py                   # CLI interface and example usage
```

### Key Files Explained

#### `config.py`
Contains configuration settings including:
- `DATABASE_URL`: Connection string for PostgreSQL
- `EMBEDDING_MODEL`: Name of the sentence-transformers model (default: "all-MiniLM-L6-v2")
- `EMBEDDING_DIMENSION`: Vector dimension for the embedding model (384 for all-MiniLM-L6-v2)

```python
# Key parts of config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pgvector_demo")

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default embedding model from sentence-transformers
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2 model
```

#### `db.py`
Handles database connection and ORM model for document storage:
- Creates connection to PostgreSQL via SQLAlchemy
- Defines Document model with vector embedding column
- Contains functions to initialize the database

```python
# Key parts of db.py
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(Text, nullable=True)
    embedding = Column(Vector(EMBEDDING_DIMENSION))

def init_db():
    """Initialize database with pgvector extension and tables."""
    # Connect directly with psycopg2 to create extension
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Create pgvector extension if it doesn't exist
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Close psycopg2 connection
    cursor.close()
    conn.close()
    
    # Create tables
    Base.metadata.create_all(bind=engine)
```

#### `embeddings.py`
Manages text embedding using sentence-transformers:
- Creates embeddings for single texts or batches of documents
- Handles conversion of different input formats to embeddings

```python
# Key parts of embeddings.py
class Embedder:
    def __init__(self, model_name=EMBEDDING_MODEL):
        """Initialize embedder with the specified model."""
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text):
        """Embed a single text into a vector."""
        embedding = self.model.encode(text)
        return embedding.astype(np.float32)
    
    def embed_documents(self, documents):
        """Embed a list of documents into vectors."""
        texts = []
        for doc in documents:
            if isinstance(doc, dict) and 'content' in doc:
                texts.append(doc['content'])
            elif isinstance(doc, str):
                texts.append(doc)
        
        embeddings = self.model.encode(texts)
        return [e.astype(np.float32) for e in embeddings]
```

#### `document_store.py`
Handles document storage and retrieval operations:
- Adds documents with metadata to the database
- Performs similarity searches using pgvector
- Converts between internal and external document representations

```python
# Key parts of document_store.py
class DocumentStore:
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
```

#### `rag.py`
Implements Retrieval Augmented Generation:
- Retrieves relevant documents from the document store
- Builds context from retrieved documents
- Generates answers using OpenAI's language models

```python
# Key parts of rag.py
class RAG:
    def __init__(self, llm_model_name="gpt-3.5-turbo"):
        """Initialize RAG with document store and language model."""
        self.document_store = DocumentStore()
        
        # Check if OpenAI API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize LLM with ChatOpenAI
        self.llm = ChatOpenAI(temperature=0.7, model_name=llm_model_name)
        
        # Define prompt template for RAG
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Answer the question based on the provided context.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
        )
        
        # Create chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def query(self, question, top_k=3):
        """Execute RAG query - retrieve documents and generate answer."""
        # Retrieve relevant documents
        retrieved_docs = self.document_store.similarity_search(question, top_k=top_k)
        
        # Format context from retrieved documents
        context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)])
        
        # Generate answer
        answer = self.chain.run(question=question, context=context)
        
        # Format sources for citation
        sources = [{"id": doc["id"], "content": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"]} 
                  for doc in retrieved_docs]
        
        return {
            "answer": answer,
            "sources": sources
        }
```

#### `main.py`
Provides command-line interface and example usage:
- CLI commands for database setup, search, and RAG queries
- Sample document creation and handling

```python
# Key parts of main.py
def setup_database():
    """Set up the database with pgvector extension."""
    init_db()
    print("Database initialized with pgvector extension.")

def add_sample_documents():
    """Add sample documents to the document store."""
    document_store = DocumentStore()
    
    sample_docs = [
        {
            "content": "PostgreSQL is a powerful, open source object-relational database system...",
            "metadata": {"source": "postgresql.org", "category": "database"}
        },
        # More sample documents...
    ]
    
    doc_ids = document_store.add_documents(sample_docs)
    print(f"Added {len(doc_ids)} sample documents with IDs: {doc_ids}")

def main():
    parser = argparse.ArgumentParser(description="pgvector RAG Demo")
    parser.add_argument("--setup", action="store_true", help="Set up database and add sample documents")
    parser.add_argument("--search", help="Search for documents similar to query")
    parser.add_argument("--query", help="Perform RAG query with question")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    
    args = parser.parse_args()
    # Handle the different commands...
```

## Quick Setup

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   cd PGVector
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the PostgreSQL with pgvector container:
   ```bash
   docker compose up -d
   ```

4. Set up environment variables:
   ```bash
   # Option 1: Set variables directly in terminal
   DATABASE_URL=postgresql://postgres:exidhani11@localhost:5433/pgvector_demo
   OPENAI_API_KEY=your_openai_key_here
   
   # Option 2: Create a .env file with these values
   ```

5. Initialize the database and add sample documents:
   ```bash
   DATABASE_URL=postgresql://postgres:exidhani11@localhost:5433/pgvector_demo python main.py --setup
   ```

## Docker Setup Explained

The system uses Docker to run PostgreSQL with the pgvector extension pre-installed:

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    container_name: pgvector-postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: exidhani11
      POSTGRES_DB: pgvector_demo
    ports:
      - "5433:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    command: postgres -c 'shared_preload_libraries=vector'

volumes:
  pgdata:
```

### Docker Configuration Details

- **Image**: Uses `pgvector/pgvector:pg15` - a pre-built image with PostgreSQL 15 and pgvector extension
- **Container Name**: Names the container `pgvector-postgres` for easy reference
- **Environment Variables**: 
  - Sets the database username to `postgres`
  - Sets the password to `exidhani11`
  - Creates a database named `pgvector_demo`
- **Port Mapping**: Maps container port 5432 to host port 5433 to avoid conflicts
- **Volume**: Creates a persistent volume `pgdata` for database storage
- **Command**: Loads the vector extension with `shared_preload_libraries=vector`

## Core Components

### Database Layer

The database layer uses PostgreSQL with the pgvector extension which adds:

- A new `vector` data type for storing embeddings
- Vector similarity operators for cosine distance (`<=>`)
- Index support for accelerating vector similarity queries

Within our system, this is managed through SQLAlchemy ORM with the following components:

- **Document Model**: Stores content, metadata, and vector embeddings
- **Vector Column**: Uses the pgvector's Column type to store embeddings
- **Database Initialization**: Ensures the pgvector extension is enabled

### Embedding Engine

The embedding engine converts text to vector representations:

- **SentenceTransformer Model**: Uses the "all-MiniLM-L6-v2" model by default
- **Embedding Creation**: Converts text to 384-dimensional vectors
- **Batch Processing**: Can efficiently embed multiple documents at once
- **Type Conversion**: Ensures the right format for database storage

### Document Store

The document store manages interactions with the database:

- **Document Addition**: Adds new documents with content, metadata, and embeddings
- **Batch Document Import**: Efficiently handles adding multiple documents
- **Vector Similarity Search**: Finds similar documents using cosine similarity
- **Raw SQL for pgvector**: Uses direct PostgreSQL commands for vector operations

### RAG Implementation

The RAG implementation combines retrieval with generation:

- **Document Retrieval**: Fetches relevant documents based on query similarity
- **Context Building**: Formats retrieved documents into context for the LLM
- **Answer Generation**: Uses OpenAI's models to generate answers from context
- **Source Attribution**: Tracks which documents contributed to the answer

## Usage Examples

### Initialize Database

Initialize the database and populate with sample documents:

```bash
DATABASE_URL=postgresql://postgres:exidhani11@localhost:5433/pgvector_demo python main.py --setup
```

Example output:
```
Database initialized with pgvector extension.
Added 5 sample documents with IDs: [1, 2, 3, 4, 5]
```

The sample documents include:
1. Information about PostgreSQL
2. Description of pgvector
3. Explanation of RAG
4. Information about embeddings
5. Details about vector databases

### Search for Similar Documents

Search for documents semantically similar to a query:

```bash
DATABASE_URL=postgresql://postgres:exidhani11@localhost:5433/pgvector_demo python main.py --search "vector databases" --top-k 3
```

Actual output:
```
Search results for query: 'vector databases'
==================================================
Result 1 [Similarity: 0.7125]
Content: Vector databases specialize in storing and querying vector embeddings efficiently, offering features like approximate nearest neighbor (ANN) search to find similar vectors quickly.
Metadata: {'source': 'database guide', 'category': 'database'}
--------------------------------------------------
Result 2 [Similarity: 0.5493]
Content: pgvector is a PostgreSQL extension for vector similarity search. It allows you to store vectors in a new data type and query them by cosine distance, Euclidean distance, or inner product.
Metadata: {'source': 'github.com/pgvector/pgvector', 'category': 'extension'}
--------------------------------------------------
Result 3 [Similarity: 0.3979]
Content: PostgreSQL is a powerful, open source object-relational database system with over 30 years of active development that has earned it a strong reputation for reliability, feature robustness, and performance.
Metadata: {'source': 'postgresql.org', 'category': 'database'}
--------------------------------------------------
```

This output shows:
- Each result's similarity score (1.0 would be perfect match)
- The full content of each matching document
- Metadata associated with each document
- Results are sorted by descending similarity

### Query with RAG

Perform a RAG query to answer a question based on retrieved documents:

```bash
DATABASE_URL=your_database_url OPENAI_API_KEY=your_openai_key_here python main.py --query "What is pgvector and how does it relate to vector databases?" --top-k 3
```

Actual output:
```
RAG response for question: 'What is pgvector and how does it relate to vector databases?'
==================================================
Answer: pgvector is a PostgreSQL extension for vector similarity search that allows storing vectors in a new data type and querying them by cosine distance, Euclidean distance, or inner product. It relates to vector databases by specializing in storing and querying vector embeddings efficiently, offering features like approximate nearest neighbor (ANN) search to find similar vectors quickly.

Sources:
1. pgvector is a PostgreSQL extension for vector similarity search. It allows you to store vectors in a...
2. PostgreSQL is a powerful, open source object-relational database system with over 30 years of active...
3. Vector databases specialize in storing and querying vector embeddings efficiently, offering features...
```

This output shows:
- A coherent answer synthesized from multiple documents
- The sources that contributed to the answer (first 100 chars shown)
- The system retrieved the most relevant documents and used them to generate the answer

## Adding Custom Documents

### Option 1: Using Python Script

```python
from document_store import DocumentStore
import os

# Connect to the database
DATABASE_URL = "your_database_url"
os.environ["DATABASE_URL"] = DATABASE_URL

# Create documents
docs = [
    {
        "content": "Machine learning models work by finding patterns in large datasets. They use algorithms to identify relationships between features and outcomes.",
        "metadata": {"source": "ML textbook", "category": "machine learning"}
    },
    {
        "content": "Neural networks are composed of layers of interconnected nodes. Each connection has a weight that adjusts as learning proceeds.",
        "metadata": {"source": "Deep Learning basics", "category": "neural networks"}
    }
]

# Add to database
doc_store = DocumentStore()
doc_ids = doc_store.add_documents(docs)
print(f"Added documents with IDs: {doc_ids}")
```

### Option 2: Using Command Line

```bash
DATABASE_URL=postgresql://postgres:exidhani11@localhost:5433/pgvector_demo python -c "from document_store import DocumentStore; ds = DocumentStore(); ds.add_document('Your document content', {'source': 'custom source'})"
```

## Troubleshooting

### Database Connection Issues

If you cannot connect to the database:

1. Verify the Docker container is running:
   ```bash
   docker ps
   ```
   Expected output:
   ```
   CONTAINER ID   IMAGE                   COMMAND                  CREATED        STATUS        PORTS                    NAMES
   b1a2c3d4e5f6   pgvector/pgvector:pg15  "docker-entrypoint.s…"   2 hours ago    Up 2 hours    0.0.0.0:5433->5432/tcp   pgvector-postgres
   ```

2. Check container logs for any issues:
   ```bash
   docker logs pgvector-postgres
   ```

3. Test the database connection:
   ```bash
   docker exec -it pgvector-postgres psql -U postgres -d pgvector_demo -c "SELECT 1;"
   ```
   Expected output:
   ```
    ?column? 
   ----------
           1
   (1 row)
   ```

### Vector Format Issues

If you encounter errors with vector formats:

1. Ensure vector data is properly formatted for PostgreSQL:
   ```
   # Correct format:
   '[0.1, 0.2, 0.3, ...]'
   
   # Incorrect formats:
   '{0.1, 0.2, 0.3, ...}'  # Uses curly braces
   '(0.1, 0.2, 0.3, ...)'  # Uses parentheses
   ```

2. Verify that vector dimensions match the configuration:
   ```python
   # In config.py
   EMBEDDING_DIMENSION = 384  # Must match actual vector dimensions
   ```

3. For direct SQL queries, ensure the vector is cast correctly:
   ```sql
   SELECT 1 - (embedding <=> '[0.1, 0.2, 0.3, ...]'::vector) as similarity 
   FROM documents
   ```

### OpenAI API Issues

If you encounter issues with RAG and OpenAI:

1. Verify your API key is set correctly
2. Check that the openai package is installed: `pip install openai`
3. Try using a different model if you're hitting rate limits

## Shutting Down

To stop the Docker container:

```bash
docker compose down
```

To stop and remove all data (including the database volume):

```bash
docker compose down -v
```

## Advanced Configuration

### Changing the Embedding Model

You can modify `config.py` to use a different sentence-transformers model:

```python
# Current configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions

# Examples of other models you could use
# EMBEDDING_MODEL = "all-mpnet-base-v2"  # 768 dimensions, more accurate
# EMBEDDING_MODEL = "all-MiniLM-L12-v2"  # 384 dimensions, better performance
```

Remember to update `EMBEDDING_DIMENSION` to match the dimension of your chosen model.

### Performance Optimization

For larger document collections, consider:

1. Adding an index to speed up vector searches:
   ```sql
   CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
   ```

2. Implementing batch processing for large document imports
3. Adding caching for frequent queries
4. Using connection pooling for concurrent usage 