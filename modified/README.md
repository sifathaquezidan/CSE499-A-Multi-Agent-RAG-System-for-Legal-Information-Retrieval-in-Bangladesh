# 🏛️ Multi-LLM Bangladeshi Legal Assistant

An advanced conversational legal RAG (Retrieval-Augmented Generation) chatbot supporting **multiple LLM providers** for enhanced flexibility and cost optimization.

## ✨ New Features

### 🤖 Multi-LLM Provider Support
- **OpenAI** (GPT-4, GPT-3.5) - Premium performance
- **Ollama** (Local LLMs) - Privacy-focused, cost-free
- **Google Gemini** - Competitive alternative

### 🔄 Dual Configuration Modes
- **Multi-Provider Mode**: Choose your preferred AI provider
- **Legacy Mode**: Backward-compatible OpenAI-only setup

### 🛠️ Enhanced Architecture
- Provider abstraction layer for easy extensibility
- Unified interface across all LLM providers
- Real-time configuration validation

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone [repository-url]
cd modified

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file:

```bash
# Required for all modes
TAVILY_API_KEY=your_tavily_api_key

# Choose your preferred provider(s)
OPENAI_API_KEY=your_openai_api_key        # For OpenAI models
GEMINI_API_KEY=your_google_gemini_key     # For Gemini models

# For Ollama (local setup)
# No API key needed - just install Ollama locally
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Configure Your Provider

1. Open the application in your browser
2. Choose configuration mode in the sidebar:
   - **🚀 Multi-Provider (New)**: Full provider selection
   - **🔧 Legacy (OpenAI Only)**: Original setup
3. Select your LLM provider and model
4. Start chatting!

## 🤖 Supported Providers

### OpenAI ⭐ (Recommended)
- **Models**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Pros**: Best performance, reliable, structured output support
- **Cons**: Requires API credits
- **Setup**: Just add your OpenAI API key

### Ollama 🏠 (Privacy-Focused)
- **Models**: Llama2, Mistral, Phi, Vicuna, CodeLlama, and more
- **Pros**: Free, runs locally, complete privacy, offline capable
- **Cons**: Requires local setup, slower than cloud APIs
- **Setup**:
  ```bash
  # Install Ollama
  curl -fsSL https://ollama.ai/install.sh | sh
  
  # Pull a model
  ollama pull llama2
  
  # Start the service
  ollama serve
  ```

### Google Gemini 🔍 (Cost-Effective)
- **Models**: Gemini-Pro, Gemini-1.5-Pro, Gemini-1.5-Flash
- **Pros**: Competitive pricing, strong reasoning, good performance
- **Cons**: Newer, less tested with legal content
- **Setup**: Get API key from [Google AI Studio](https://aistudio.google.com/)

## 📊 Feature Comparison

| Feature | OpenAI | Ollama | Gemini |
|---------|--------|--------|--------|
| Cost | API usage | Free* | API usage |
| Privacy | Cloud | Local | Cloud |
| Speed | Fast | Moderate | Fast |
| Offline | No | Yes | No |
| Setup Complexity | Low | Medium | Low |
| Legal Performance | Excellent | Good | Good |

*Requires local computational resources

## 🏗️ System Architecture

### Two-Stage Intelligence System

1. **Clarification Agent** 📝
   - Analyzes queries for completeness
   - Asks targeted follow-up questions
   - Synthesizes clear legal questions
   - Supports all LLM providers

2. **RAG Agent** 🧠
   - Searches 22+ Bangladeshi legal documents
   - Grades document relevance
   - Falls back to web search when needed
   - Cites sources and provides detailed answers

### Provider Abstraction Layer

```python
# Unified interface for all providers
llm_manager = LLMManager(config)
chat_model = llm_manager.get_chat_model()
structured_model = llm_manager.get_structured_output_model(MyModel)
```

## 📚 Legal Document Coverage

- **Constitutional Law**: Bangladesh Constitution
- **Criminal Law**: Penal Code, CrPC, Criminal Law Amendments  
- **Civil Law**: Code of Civil Procedure, Citizenship Act
- **Land Law**: State Acquisition & Tenancy Act, Land Reforms
- **Commercial Law**: Negotiable Instruments Act
- **Social Law**: Domestic Violence, RTI Act
- **Administrative Law**: Laws Revision & Declaration Acts

## 🧪 Testing

Test your provider setup:

```bash
python test_providers.py
```

This script will:
- Validate all configured providers
- Test basic chat functionality
- Check structured output support
- Report configuration issues

## 📖 Usage Examples

### Example Queries

**Property Law:**
```
"What documents are needed for land registration under the State Acquisition and Tenancy Act in Bangladesh?"
```

**Criminal Law:**
```
"What is the bail procedure for non-bailable offenses under Section 497 of CrPC?"
```

**Family Law:**
```
"What are the grounds for divorce under Muslim Family Laws Ordinance in Bangladesh?"
```

### Provider Selection Guidance

- **For production/critical use**: OpenAI (most reliable)
- **For privacy-sensitive queries**: Ollama (local processing)
- **For cost optimization**: Gemini (competitive pricing)
- **For experimentation**: Try all providers and compare

## 🔧 Configuration Options

### Temperature Settings
- **0.0**: Deterministic, factual responses
- **0.3**: Slightly more creative
- **0.7+**: More creative, less predictable

### Model Selection Tips
- **GPT-4o-mini**: Fast, cost-effective for most queries
- **GPT-4o**: Maximum accuracy for complex legal questions
- **Llama2**: Good general performance for local usage
- **Gemini-Pro**: Balanced performance and cost

## 🔄 Migration from Previous Version

**Existing users**: No changes required! Use "Legacy Mode" for identical functionality.

**New features**: Switch to "Multi-Provider Mode" to access new capabilities.

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions.

## 🛠️ Development

### Adding New Providers

1. Implement `BaseLLMProvider` interface
2. Add to `LLMProviderFactory`
3. Update UI configuration
4. Test with `test_providers.py`

### File Structure
```
├── app.py                 # Main Streamlit application
├── llm_providers.py       # Provider abstraction layer
├── config_manager.py      # Configuration management
├── test_providers.py      # Provider testing utilities
├── requirements.txt       # Dependencies
├── MIGRATION_GUIDE.md     # Migration instructions
└── data/                  # Legal documents (PDFs)
```

## 🐛 Troubleshooting

### Common Issues

**Ollama connection failed:**
- Ensure Ollama is running: `ollama serve`
- Check if model is downloaded: `ollama list`
- Verify base URL: `http://localhost:11434`

**Gemini API errors:**
- Verify API key in Google AI Studio
- Check quota and billing settings
- Ensure correct model names

**Import errors:**
- Update dependencies: `pip install -r requirements.txt`
- Check Python version compatibility

**Performance issues:**
- Try different models within the same provider
- Adjust temperature settings
- Consider switching providers for specific use cases

## 📊 Performance Monitoring

The application provides real-time feedback on:
- Provider initialization status
- Model loading times
- Query processing speed
- Error rates and recovery

## 🔒 Security & Privacy

### Data Handling
- **OpenAI/Gemini**: Queries sent to cloud APIs
- **Ollama**: All processing happens locally
- **Vector DB**: Stored locally (ChromaDB)
- **Chat History**: Maintained in session state only

### API Key Security
- Environment variables only
- Never logged or displayed
- Secure input fields in UI

## 📞 Support

1. Check [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
2. Run diagnostic test: `python test_providers.py`
3. Review application logs in terminal
4. Verify all dependencies and API keys

## 🎯 Roadmap

- [ ] Add Azure OpenAI support
- [ ] Implement Claude (Anthropic) provider
- [ ] Add model performance benchmarking
- [ ] Support for custom local models
- [ ] Enhanced structured output for non-OpenAI providers
- [ ] Multi-provider ensemble responses

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com/) - LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration  
- [Streamlit](https://streamlit.io/) - Web interface
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Tavily](https://tavily.com/) - Web search API

---

**⚖️ Legal Disclaimer**: This tool provides informational content only and should not be considered as professional legal advice. Always consult qualified legal professionals for actual legal matters.
