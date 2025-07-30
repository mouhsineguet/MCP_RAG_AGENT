import streamlit as st
import asyncio
import tempfile
import os
from typing import List, Dict, Any
import json
import logging

from ollama_client import OllamaClient, RAGPipeline
from mcp_client import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Document Chat AI",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = None
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

async def initialize_clients():
    """Initialize Ollama and MCP clients"""
    try:
        # Initialize Ollama client
        ollama_client = OllamaClient(model="llama3.2:3b")
        
        # Check if model is available
        if not await ollama_client.check_model_available():
            st.warning("Llama 3.2 3B model not found. Attempting to pull...")
            if await ollama_client.pull_model():
                st.success("Model pulled successfully!")
            else:
                st.error("Failed to pull model. Please ensure Ollama is running.")
                return None, None, None
        
        # Initialize MCP client
        mcp_client = MCPClient()
        await mcp_client.connect()
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(ollama_client, mcp_client)
        
        return ollama_client, mcp_client, rag_pipeline
        
    except Exception as e:
        st.error(f"Failed to initialize clients: {e}")
        return None, None, None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

async def process_document(file_path: str, mcp_client: MCPClient):
    """Process uploaded document"""
    try:
        result = await mcp_client.call_tool("upload_document", {"file_path": file_path})
        return result
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

async def chat_with_document(user_input: str, rag_pipeline: RAGPipeline):
    """Chat with the document using RAG pipeline"""
    try:
        # Prepare messages for the model
        messages = []
        for msg in st.session_state.messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        response = await rag_pipeline.search_and_generate(
            query=user_input,
            messages=messages
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return f"Error processing your message: {str(e)}"

def main():
    st.title("📄 Document Chat AI")
    st.markdown("Upload a document and chat with it using AI!")
    
    # Initialize clients if not already done
    if (st.session_state.ollama_client is None or 
        st.session_state.mcp_client is None or 
        st.session_state.rag_pipeline is None):
        
        with st.spinner("Initializing AI clients..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                ollama_client, mcp_client, rag_pipeline = loop.run_until_complete(
                    initialize_clients()
                )
                
                if ollama_client and mcp_client and rag_pipeline:
                    st.session_state.ollama_client = ollama_client
                    st.session_state.mcp_client = mcp_client
                    st.session_state.rag_pipeline = rag_pipeline
                    st.success("AI clients initialized successfully!")
                else:
                    st.error("Failed to initialize AI clients. Please check your setup.")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Error initializing clients: {e}")
                st.stop()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'docx', 'txt'],
            help="Upload a PDF, DOCX, or TXT file to chat with"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Save uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    if file_path:
                        # Process document
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        result = loop.run_until_complete(
                            process_document(file_path, st.session_state.mcp_client)
                        )
                        
                        # Clean up temp file
                        os.unlink(file_path)
                        
                        if result["success"]:
                            st.success(result["message"])
                            st.session_state.document_loaded = True
                        else:
                            st.error(result["message"])
        
        # Document status
        if st.session_state.document_loaded:
            st.success("✅ Document loaded and ready for chat!")
        else:
            st.warning("⚠️ Please upload and process a document first")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat with Your Document")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your document..."):
        if not st.session_state.document_loaded:
            st.warning("Please upload and process a document first!")
            st.stop()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                response = loop.run_until_complete(
                    chat_with_document(prompt, st.session_state.rag_pipeline)
                )
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Powered by:** Ollama (Llama 3.2 4B) • FastMCP • Streamlit • Sentence Transformers"
    )

if __name__ == "__main__":
    main()