import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import ollama

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host)
        
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              context: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1000) -> str:
        """
        Generate a response using Ollama
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            context: Optional context from document search
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response string
        """
        try:
            # Prepare the system prompt
            system_prompt = """You are a helpful AI assistant that can answer questions about uploaded documents. 
            When provided with context from a document, use that information to answer questions accurately.
            If you don't have enough context to answer a question, say so clearly.
            Always be helpful, accurate, and cite the document when relevant."""
            
            # Add context to the system prompt if provided
            if context:
                system_prompt += f"\n\nDocument Context:\n{context}"
            
            # Prepare messages for Ollama
            ollama_messages = [{"role": "system", "content": system_prompt}]
            ollama_messages.extend(messages)
            
            # Generate response
            response = self.client.chat(
                model=self.model,
                messages=ollama_messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def check_model_available(self) -> bool:
        """Check if the model is available"""
        try:
            models = self.client.list()
            
            model_names = [model['model'] for model in models['models']]
            return self.model in model_names
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    async def pull_model(self) -> bool:
        """Pull the model if not available"""
        try:
            self.client.pull(self.model)
            return True
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False

class RAGPipeline:
    def __init__(self, ollama_client: OllamaClient, mcp_client=None):
        self.ollama_client = ollama_client
        self.mcp_client = mcp_client
        
    async def search_and_generate(self, 
                                 query: str, 
                                 messages: List[Dict[str, str]],
                                 k: int = 5) -> str:
        """
        Search for relevant context and generate response
        
        Args:
            query: User query
            messages: Chat history
            k: Number of search results
            
        Returns:
            Generated response
        """
        try:
            # Search for relevant context using MCP
            if self.mcp_client:
                search_result = await self.mcp_client.call_tool("search_document", {"query": query, "k": k})
                
                if search_result.get("success"):
                    # Extract relevant chunks
                    results = search_result.get("results", [])
                    context_chunks = [result["chunk"] for result in results[:3]]  # Top 3 most relevant
                    context = "\n\n".join(context_chunks)
                    
                    # Generate response with context
                    response = await self.ollama_client.generate_response(
                        messages=messages,
                        context=context
                    )
                    return response
                else:
                    # No context available, generate response without context
                    response = await self.ollama_client.generate_response(messages=messages)
                    return response
            else:
                # No MCP client, generate response without context
                response = await self.ollama_client.generate_response(messages=messages)
                return response
                
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return f"Error processing your request: {str(e)}"