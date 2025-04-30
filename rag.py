import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from document_store import DocumentStore

class RAG:
    def __init__(self, llm_model_name="gpt-3.5-turbo"):
        """Initialize RAG with document store and language model."""
        self.document_store = DocumentStore()
        
        # Check if OpenAI API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize LLM with ChatOpenAI instead of OpenAI
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
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": []
            }
        
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
    
    def add_document(self, content, metadata=None):
        """Add a document to the RAG system."""
        return self.document_store.add_document(content, metadata)
    
    def add_documents(self, documents):
        """Add multiple documents to the RAG system."""
        return self.document_store.add_documents(documents) 