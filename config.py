import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pgvector_demo")

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default embedding model from sentence-transformers
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2 model 