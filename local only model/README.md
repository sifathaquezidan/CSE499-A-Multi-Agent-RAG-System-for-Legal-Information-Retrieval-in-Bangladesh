# Bangladeshi Legal Assistant - Local LLM Version

A conversational legal assistant for Bangladeshi law powered entirely by local LLMs via Ollama. Features privacy-first design with optional OpenAI embeddings for enhanced performance.

## 🔄 Changes from Original Version

### ✅ Completed Modifications:

- **Replaced all OpenAI LLMs with Ollama** - No more remote API calls for language models
- **Added UI model selection** - Choose from multiple Ollama models (llama3.1, mistral, etc.)
- **Hybrid embedding support** - Use OpenAI embeddings (optional) or Ollama embeddings (fallback)
- **Privacy-first approach** - All LLM processing happens locally on your machine
- **Updated requirements** - Added langchain-ollama and ollama dependencies

### 🚀 New Features:

- **Model Selection Dropdown** - Easy switching between Ollama models
- **Embedding Choice** - Toggle between OpenAI and Ollama embeddings
- **Local Processing Indicators** - Clear messaging about local vs remote processing
- **Enhanced Privacy** - No data sent to external APIs for LLM operations

## 🛠️ Installation & Setup

### Prerequisites

1. **Install Ollama** (Required for LLM functionality)

   ```bash
   # Download and install from: https://ollama.ai/
   # Or using package managers:

   # Windows (using winget)
   winget install ollama

   # macOS (using brew)
   brew install ollama

   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull Required Models**

   ```bash
   # Core model (recommended for most users)
   ollama pull llama3.1

   # Alternative models (optional)
   ollama pull llama3.1:8b      # Smaller, faster
   ollama pull llama3.1:70b     # Larger, more capable
   ollama pull mistral          # Alternative model
   ollama pull gemma2           # Google's model
   ```

3. **Pull Embedding Model** (if not using OpenAI embeddings)
   ```bash
   ollama pull nomic-embed-text
   ```

### Python Environment Setup

1. **Clone/Download the project**

   ```bash
   git clone <repository-url>
   cd "langGraph agentic RAG/local/local only model"
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (Optional)
   Create a `.env` file:

   ```env
   # Optional: Only if using OpenAI embeddings
   OPENAI_API_KEY=your_openai_api_key_here

   # Required: For web search fallback
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## 🚀 Running the Application

### Start Ollama Service

Ensure Ollama is running (usually starts automatically after installation):

```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve
```

### Launch the Legal Assistant

```bash
streamlit run app.py
```

## 🔧 Configuration Options

### Model Selection

- **llama3.1** (Recommended) - Best balance of performance and speed
- **llama3.1:8b** - Faster, good for lower-end hardware
- **llama3.1:70b** - Highest quality, requires powerful hardware
- **mistral** - Alternative high-quality model
- **gemma2** - Google's efficient model

### Embedding Options

1. **OpenAI Embeddings** (Recommended if available)

   - Higher quality semantic search
   - Requires OpenAI API key
   - Small cost per embedding

2. **Ollama Embeddings** (Privacy-focused)
   - Completely local processing
   - No external API calls
   - Free but potentially lower quality

## 📊 Performance Considerations

### Hardware Requirements

- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB+ RAM, GPU support
- **Optimal**: 32GB+ RAM, dedicated GPU

### Model Performance vs Resource Usage

| Model        | Size   | RAM Usage | Speed  | Quality   |
| ------------ | ------ | --------- | ------ | --------- |
| llama3.1:8b  | ~4.7GB | 8GB+      | Fast   | Good      |
| llama3.1     | ~4.7GB | 8GB+      | Medium | Very Good |
| llama3.1:70b | ~40GB  | 64GB+     | Slow   | Excellent |
| mistral      | ~4.1GB | 8GB+      | Fast   | Very Good |

## 🔒 Privacy & Security

### Data Privacy

- **LLM Processing**: 100% local via Ollama
- **Document Storage**: Local ChromaDB database
- **Embeddings**: Optional - can be fully local
- **Web Search**: Only when local documents insufficient

### API Usage

- **OpenAI**: Only for embeddings (optional)
- **Tavily**: Only for web search fallback
- **No other external dependencies**

## 📁 Project Structure

```
local only model/
├── app.py                    # Main Streamlit application
├── test.py                   # Alternative/test version
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .env                     # Environment variables (create this)
├── data/                    # PDF documents (22+ legal files)
├── chroma_db/              # Vector database storage
└── __pycache__/            # Python cache files
```

## 🐛 Troubleshooting

### Common Issues

1. **"Connection error to Ollama"**

   ```bash
   # Check if Ollama is running
   ollama list

   # Restart Ollama if needed
   ollama serve
   ```

2. **"Model not found"**

   ```bash
   # Pull the required model
   ollama pull llama3.1
   ```

3. **High memory usage**

   - Use smaller models (llama3.1:8b)
   - Close other applications
   - Consider cloud deployment

4. **Slow responses**
   - Use faster models
   - Reduce document chunk size
   - Enable GPU acceleration in Ollama

### Performance Optimization

1. **Enable GPU Support** (if available)

   ```bash
   # Ollama automatically detects and uses GPU
   # Verify GPU usage:
   ollama list
   ```

2. **Adjust Model Parameters**
   - Lower temperature for more focused responses
   - Adjust context window for longer documents

## 🔄 Migration from Original Version

If upgrading from the OpenAI-only version:

1. **Install Ollama and models** (see setup above)
2. **Update requirements**: `pip install -r requirements.txt`
3. **Optional**: Keep existing vector database
4. **Update configuration** in the UI
5. **Test with smaller models first**

## 📈 Future Enhancements

Planned improvements:

- [ ] Support for more Ollama models
- [ ] Advanced model configuration options
- [ ] Performance monitoring dashboard
- [ ] Document upload via UI
- [ ] Multi-language support
- [ ] Export conversation history

## 🤝 Contributing

Contributions welcome! Areas of focus:

- Model performance optimization
- New Ollama model integrations
- UI/UX improvements
- Documentation enhancements

## 📄 License

[Add your license information here]

---

**Note**: This application is for informational purposes only and does not constitute legal advice. Always consult with qualified legal professionals for actual legal matters.
