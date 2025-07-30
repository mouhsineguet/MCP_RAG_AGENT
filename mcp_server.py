import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import tempfile
import os

from fastmcp import FastMCP
from fastmcp.tools import tool
from sentence_transformers import SentenceTransformer
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
import faiss
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = []
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Chunk text into smaller pieces with overlap"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        embeddings = self.embedding_model.encode(chunks)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for similarity search"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks to query"""
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "index": int(idx)
                })
        
        return results
    
    def process_document(self, file_path: str) -> bool:
        """Process a document and build search index"""
        try:
            # Extract text based on file extension
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_ext == '.docx':
                text = self.extract_text_from_docx(file_path)
            elif file_ext == '.txt':
                text = self.extract_text_from_txt(file_path)
            else:
                logger.error(f"Unsupported file type: {file_ext}")
                return False
            
            if not text.strip():
                logger.error("No text extracted from document")
                return False
            
            # Chunk the text
            self.chunks = self.chunk_text(text)
            
            # Create embeddings
            embeddings = self.create_embeddings(self.chunks)
            self.embeddings = embeddings
            
            # Build search index
            self.build_index(embeddings)
            
            logger.info(f"Document processed: {len(self.chunks)} chunks created")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return False

# Global document processor instance
doc_processor = DocumentProcessor()

# Initialize FastMCP
mcp = FastMCP("document-chat-agent")

@mcp.tool
async def upload_document(file_path: str) -> Dict[str, Any]:
    """
    Upload and process a document for chat.
    
    Args:
        file_path: Path to the document file (PDF, DOCX, or TXT)
    
    Returns:
        Dict with success status and message
    """
    try:
        if not os.path.exists(file_path):
            return {"success": False, "message": "File not found"}
        
        success = doc_processor.process_document(file_path)
        
        if success:
            return {
                "success": True,
                "message": f"Document processed successfully. {len(doc_processor.chunks)} chunks created."
            }
        else:
            return {"success": False, "message": "Failed to process document"}
            
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@mcp.tool
async def search_document(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Search for relevant information in the uploaded document.
    
    Args:
        query: The search query
        k: Number of results to return (default 5)
    
    Returns:
        Dict with search results
    """
    try:
        if doc_processor.index is None:
            return {"success": False, "message": "No document loaded"}
        
        results = doc_processor.search_similar(query, k)
        
        return {
            "success": True,
            "results": results,
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error searching document: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@mcp.tool
async def get_document_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded document.
    
    Returns:
        Dict with document information
    """
    try:
        if doc_processor.index is None:
            return {"success": False, "message": "No document loaded"}
        
        return {
            "success": True,
            "info": {
                "total_chunks": len(doc_processor.chunks),
                "embedding_dimension": doc_processor.embeddings.shape[1] if doc_processor.embeddings is not None else 0,
                "model_name": doc_processor.embedding_model.get_sentence_embedding_dimension()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()