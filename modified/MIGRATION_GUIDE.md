# Multi-LLM Provider System - Migration Guide

## Overview

The Legal RAG application has been enhanced to support multiple LLM providers while maintaining full backward compatibility. You can now choose between:

- **OpenAI** (GPT-4, GPT-3.5-turbo models)
- **Ollama** (Local LLMs like Llama2, Mistral, etc.)
- **Google Gemini** (Gemini Pro models)

## 🚀 New Features

### 1. Provider Abstraction Layer
- Unified interface for all LLM providers
- Easy switching between providers
- Consistent behavior across different models

### 2. Enhanced UI
- Provider selection dropdown
- Model selection for each provider
- Real-time configuration validation
- Legacy mode for existing users

### 3. Extensible Architecture
- Easy to add new providers
- Modular design
- Clean separation of concerns

## 📋 Requirements

### Updated Dependencies

Add these to your `requirements.txt`:

```
# Multi-LLM Provider Support
langchain-google-genai  # For Google Gemini API support
google-generativeai     # Additional Gemini dependencies
```

### API Keys

Depending on which providers you want to use:

```bash
# .env file
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
GEMINI_API_KEY=your_gemini_key_here  # New
```

## 🔧 Migration Options

### Option 1: Keep Using OpenAI (Recommended for existing users)

**No changes required!** The application includes a "Legacy Mode" that works exactly like before.

1. Start the application
2. Select "🔧 Legacy (OpenAI Only)" in the sidebar
3. Enter your OpenAI and Tavily API keys as before

### Option 2: Upgrade to Multi-Provider System

1. Install new dependencies:
   ```bash
   pip install langchain-google-genai google-generativeai
   ```

2. Start the application
3. Select "🚀 Multi-Provider (New)" in the sidebar
4. Choose your preferred LLM provider
5. Configure the selected provider

## 🤖 Provider-Specific Setup

### OpenAI Setup
- Requires: OpenAI API key
- Models: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- Best for: Reliable performance, structured output support

### Ollama Setup (Local LLMs)
1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama2`
3. Start Ollama service: `ollama serve`
4. In the app, select Ollama provider
5. Choose your downloaded model

**Benefits:**
- No API costs
- Data stays local
- No internet required after setup

### Google Gemini Setup
1. Get Google AI Studio API key: https://aistudio.google.com/
2. Add `GEMINI_API_KEY` to your `.env` file
3. Select Gemini provider in the app
4. Choose from available Gemini models

## 🧪 Testing Your Setup

Use the included test script:

```bash
python test_providers.py
```

This will test all configured providers and report any issues.

## 🔄 Configuration Modes

### Multi-Provider Mode (New)
- Full provider selection
- Model-specific settings
- Advanced configuration options
- Real-time validation

### Legacy Mode
- OpenAI-only (backward compatible)
- Simple API key input
- Identical behavior to previous version

## 📊 Performance Considerations

### OpenAI
- ✅ Fastest response times
- ✅ Best structured output support
- ✅ Most reliable
- ❌ Costs per API call

### Ollama (Local)
- ✅ Free after setup
- ✅ Complete privacy
- ✅ Works offline
- ❌ Slower than cloud APIs
- ❌ Requires local resources

### Gemini
- ✅ Competitive pricing
- ✅ Good performance
- ✅ Strong reasoning capabilities
- ❌ Less tested with legal content

## 🐛 Troubleshooting

### "Provider not found" error
- Ensure all dependencies are installed
- Check import statements

### Ollama connection failed
- Verify Ollama is running: `ollama list`
- Check the base URL (default: http://localhost:11434)
- Ensure the model is downloaded

### Gemini API errors
- Verify API key is correct
- Check quota limits in Google AI Studio
- Ensure the model name is correct

### Structured output not working
- Some providers have limited structured output support
- The app will gracefully fall back to regular output parsing

## 💡 Best Practices

1. **Start with OpenAI** if you're new to the system
2. **Use Ollama** for privacy-sensitive applications
3. **Try Gemini** for cost-effective usage
4. **Test thoroughly** with your specific use cases
5. **Monitor performance** across different providers

## 🔧 Development Notes

### Adding New Providers

1. Create a new provider class in `llm_providers.py`:
   ```python
   class NewProvider(BaseLLMProvider):
       def get_chat_model(self):
           # Implementation
           pass
   ```

2. Register in the factory:
   ```python
   _providers = {
       LLMProvider.NEW_PROVIDER: NewProvider,
   }
   ```

3. Update the UI configuration

### Code Structure

- `llm_providers.py`: Provider abstraction layer
- `config_manager.py`: Configuration management
- `app.py`: Main application (updated for multi-provider)
- `test_providers.py`: Testing utilities

## 📞 Support

If you encounter issues:

1. Check this migration guide
2. Run the test script
3. Review the application logs
4. Ensure all dependencies are installed
5. Verify API keys are correctly set

## 🎯 Next Steps

After migration:

1. Test with your typical queries
2. Compare performance across providers
3. Adjust temperature and other settings
4. Consider provider-specific optimizations
5. Update your deployment scripts if needed

---

**Note:** The legacy mode ensures 100% backward compatibility. You can always fall back to the original OpenAI-only configuration if needed.
