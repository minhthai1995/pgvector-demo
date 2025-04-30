import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import numpy as np

from config import DATABASE_URL, EMBEDDING_DIMENSION

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define Document model with vector embeddings
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
    
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close() 