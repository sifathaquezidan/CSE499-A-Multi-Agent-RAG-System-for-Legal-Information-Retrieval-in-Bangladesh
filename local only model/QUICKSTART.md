# Quick Start Guide - Bangladeshi Legal Assistant (Local LLM)

Get up and running in 10 minutes! 🚀

## 1. Install Ollama (5 minutes)

### Windows

```cmd
winget install ollama
```

Or download from [ollama.ai](https://ollama.ai/)

### macOS

```bash
brew install ollama
```

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## 2. Download AI Models (3-5 minutes)

Open terminal/command prompt:

```bash
# Essential models
ollama pull llama3.1
ollama pull nomic-embed-text

# Optional: Faster model for slower computers
ollama pull llama3.1:8b
```

## 3. Install Python Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

## 4. Setup API Keys (1 minute)

Create `.env` file:

```env
# Required for web search
TAVILY_API_KEY=your_tavily_key_here

# Optional for better embeddings
OPENAI_API_KEY=your_openai_key_here
```

## 5. Launch Application (30 seconds)

```bash
streamlit run app.py
```

## 🎯 That's it!

The app will open in your browser at `http://localhost:8501`

### First Time Setup

1. Select your preferred Ollama model (start with `llama3.1`)
2. Choose embedding preference (OpenAI if you have a key, otherwise Ollama)
3. Enter your API keys in the sidebar
4. Start asking legal questions!

### Example Questions to Try

- "What are the procedures for filing a bail application in Bangladesh?"
- "What documents are needed for property registration?"
- "What are the penalties for cheque dishonour under Section 138?"

---

**Need help?** Check the full [README.md](README.md) for detailed instructions and troubleshooting.
