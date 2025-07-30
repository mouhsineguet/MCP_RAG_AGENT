# Document Chat AI

A Streamlit-based application that lets you upload documents (PDF, DOCX, TXT) and chat with them using a Retrieval-Augmented Generation (RAG) pipeline powered by Ollama (Llama 3.2) and FastMCP.

## Features

- **Document Upload:** Supports PDF, DOCX, and TXT files.
- **Semantic Search:** Uses sentence-transformer embeddings and FAISS for fast, relevant chunk retrieval.
- **Conversational AI:** Chat with your document using Llama 3.2 via Ollama.
- **RAG Pipeline:** Answers are generated using both your question and the most relevant document context.
- **Streamlit UI:** Simple, interactive web interface.

## Architecture

- **Streamlit Frontend:** User interface for uploading documents and chatting.
- **Ollama Client:** Handles communication with the Ollama LLM server.
- **MCP Server/Client:** Manages document processing, chunking, embedding, and semantic search.
- **RAG Pipeline:** Orchestrates context retrieval and LLM response generation.


```bash
pip install -r requirements.txt
```

## Getting Started

1. **Start Ollama**  
   Make sure Ollama is running locally and the required model is available:
   ```bash
   ollama run llama3.2:3b
   ```

2. **Start the MCP Server**  
   The server will be started automatically by the app, but you can also run it manually:
   ```bash
   python mcp_server.py
   ```

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

4. **Upload a Document**
   - Use the sidebar to upload a PDF, DOCX, or TXT file.
   - Click "Process Document" to index it for semantic search.

5. **Chat with Your Document**
   - Type your questions in the chat input.
   - The AI will answer using both your question and the most relevant parts of your document.

## File Structure

```
MCP/
├── app.py              # Streamlit frontend
├── mcp_client.py       # Client for communicating with the MCP server
├── mcp_server.py       # Document processing, chunking, embedding, and search
├── ollama_client.py    # Client for interacting with Ollama LLM
├── requirements.txt    # Python dependencies
├── 2503.23278v2.pdf    # Example document (optional)
```

## Customization

- **Change LLM Model:**  
  Edit the `model` parameter in `OllamaClient` (in `ollama_client.py` and `app.py`).
- **Chunk Size/Overlap:**  
  Adjust in `DocumentProcessor.chunk_text()` in `mcp_server.py`.
- **Embedding Model:**  
  Change the model name in `DocumentProcessor`'s constructor.

## Troubleshooting

- **Ollama not running:**  
  Ensure Ollama is started and the model is pulled.
- **Model not found:**  
  The app will attempt to pull the model if missing.
- **Document not loading:**  
  Check logs for errors in document parsing or embedding.

## License

MIT License

---

**Powered by:**  
- Ollama (Llama 3.2)  
- FastMCP  
- Streamlit  
- Sentence Transformers  
- FAISS 