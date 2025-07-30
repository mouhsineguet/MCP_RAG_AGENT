import asyncio
import json
import logging
from typing import Dict, Any, Optional
import subprocess
import os
import signal

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self, server_script: str = "mcp_server.py"):
        self.server_script = server_script
        self.server_process = None
        
    async def connect(self):
        """Start the MCP server process"""
        try:
            # Start the MCP server as a subprocess
            self.server_process = subprocess.Popen(
                ["python", self.server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if self.server_process.poll() is None:
                logger.info("MCP server started successfully")
                return True
            else:
                logger.error("MCP server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Result from the tool call
        """
        try:
            # Since we're using FastMCP, we'll simulate the tool calls
            # In a real implementation, this would use the MCP protocol
            
            if tool_name == "upload_document":
                return await self._upload_document(arguments.get("file_path"))
            elif tool_name == "search_document":
                return await self._search_document(
                    arguments.get("query"),
                    arguments.get("k", 5)
                )
            elif tool_name == "get_document_info":
                return await self._get_document_info()
            else:
                return {"success": False, "message": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def _upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload and process a document"""
        # Import here to avoid circular imports
        from mcp_server import doc_processor
        
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
    
    async def _search_document(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search for relevant information in the document"""
        from mcp_server import doc_processor
        
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
    
    async def _get_document_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded document"""
        from mcp_server import doc_processor
        
        try:
            if doc_processor.index is None:
                return {"success": False, "message": "No document loaded"}
            
            return {
                "success": True,
                "info": {
                    "total_chunks": len(doc_processor.chunks),
                    "embedding_dimension": doc_processor.embeddings.shape[1] if doc_processor.embeddings is not None else 0,
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def disconnect(self):
        """Stop the MCP server process"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            except Exception as e:
                logger.error(f"Error stopping MCP server: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.disconnect()