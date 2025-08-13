# Configuration Options for Bangladeshi Legal Assistant

## Ollama Model Options

### Recommended Models

- `llama3.1` - Best balance of performance and speed (default)
- `llama3.1:8b` - Faster, good for lower-end hardware
- `mistral` - Alternative high-quality model
- `gemma2` - Efficient Google model

### Advanced Models (High Memory Requirements)

- `llama3.1:70b` - Highest quality, requires 64GB+ RAM
- `codellama` - Specialized for code-related legal queries

### To pull a new model:

```bash
ollama pull model_name
```

## Embedding Model Options

### OpenAI Embeddings (Recommended)

- **Model**: `text-embedding-3-large`
- **Pros**: Higher quality semantic search, better retrieval
- **Cons**: Requires API key, small cost per embedding
- **Usage**: Set `use_openai_embeddings=True` and provide API key

### Ollama Embeddings (Privacy-First)

- **Model**: `nomic-embed-text`
- **Pros**: Completely local, no external API calls, free
- **Cons**: Potentially lower quality than OpenAI
- **Usage**: Set `use_openai_embeddings=False` or no OpenAI key

## Performance Tuning

### Hardware Requirements by Model

| Model        | Min RAM | Recommended RAM | CPU Cores | GPU VRAM |
| ------------ | ------- | --------------- | --------- | -------- |
| llama3.1:8b  | 8GB     | 12GB            | 4+        | 6GB+     |
| llama3.1     | 8GB     | 16GB            | 6+        | 8GB+     |
| llama3.1:70b | 64GB    | 128GB           | 16+       | 40GB+    |
| mistral      | 8GB     | 12GB            | 4+        | 6GB+     |
| gemma2       | 6GB     | 10GB            | 4+        | 4GB+     |

### Retrieval Configuration

```python
# In app.py, you can adjust these constants:
MAX_DOCS_PER_QUERY = 5          # Number of documents to retrieve
MINIMUN_RETRIVAL_SCORE = 0.1    # Minimum similarity threshold
MAX_QUERY_CLARIFICATION_TURNS = 5  # Max clarification rounds
EMBEDDING_BATCH_SIZE = 200      # Batch size for document embedding
```

### Temperature Settings

- **Assessment LLM**: 0 (deterministic for consistency)
- **Question Generation**: 0.3 (slight creativity for questions)
- **Document Grading**: 0 (deterministic for reliability)
- **Answer Generation**: 0.3 (balanced creativity)

## API Configuration

### Required APIs

- **Tavily API**: Web search fallback when local documents insufficient
  - Get key: https://tavily.com/
  - Used only for web search

### Optional APIs

- **OpenAI API**: Enhanced embeddings only
  - Get key: https://platform.openai.com/
  - Used only for document embeddings, not LLM inference

## Directory Structure Requirements

```
project/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── .env                  # API keys (create this)
├── data/                 # PDF documents (must exist)
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── chroma_db/           # Vector database (auto-created)
│   └── ...
└── __pycache__/         # Python cache (auto-created)
```

## Environment Variables Template

Create `.env` file:

```env
# Optional: OpenAI API key for embeddings only
OPENAI_API_KEY=sk-...

# Required: Tavily API key for web search
TAVILY_API_KEY=tvly-...

# Optional: Custom Ollama server URL (defaults to localhost:11434)
# OLLAMA_BASE_URL=http://localhost:11434

# Optional: Custom model defaults
# DEFAULT_OLLAMA_MODEL=llama3.1
# DEFAULT_EMBEDDING_MODEL=nomic-embed-text
```

## Advanced Customization

### Custom Prompts

You can modify the system prompts in `app.py`:

- `SYS_PROMPT_GRADER` - Document relevance assessment
- `SYS_PROMPT_RAG` - Answer generation
- `SYS_WEB_SEARCH_PROMPT` - Web search query rewriting
- `SYS_INITIAL_QUERY_PROMPT` - Query optimization

### Custom Model Parameters

In the model initialization, you can adjust:

```python
ChatOllama(
    model=ollama_model,
    temperature=0,
    # Additional parameters:
    # top_p=0.9,
    # top_k=40,
    # repeat_penalty=1.1,
    # num_ctx=4096,  # context window
    # num_predict=512,  # max tokens to generate
)
```

## Troubleshooting Common Issues

### Model Loading Issues

```bash
# Check available models
ollama list

# Pull missing models
ollama pull llama3.1
ollama pull nomic-embed-text

# Check Ollama status
ollama ps
```

### Memory Issues

- Use smaller models (`llama3.1:8b` instead of `llama3.1`)
- Reduce `MAX_DOCS_PER_QUERY` and `EMBEDDING_BATCH_SIZE`
- Close other applications
- Consider using swap/virtual memory

### Performance Issues

- Enable GPU acceleration (automatic with Ollama)
- Use SSD storage for vector database
- Increase system RAM
- Use faster models for development/testing

## Security Considerations

### Data Privacy

- All LLM processing happens locally
- Documents stored in local ChromaDB
- Only web search uses external APIs
- Optional OpenAI embeddings are one-way (no model fine-tuning)

### API Key Security

- Store keys in `.env` file (never commit to version control)
- Use environment variables in production
- Rotate keys regularly
- Monitor API usage for unexpected activity
