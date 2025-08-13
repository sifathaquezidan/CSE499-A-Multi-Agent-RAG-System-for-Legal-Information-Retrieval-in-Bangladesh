# Troubleshooting Guide: Common Ollama Integration Issues

## Issue: 'NoneType' object has no attribute 'status'

### Problem Description

This error occurs when Ollama's structured output functionality returns `None` instead of the expected Pydantic model object. This is a known limitation with some Ollama models that don't fully support structured output.

### Root Cause

- Ollama's `with_structured_output()` method is less reliable than OpenAI's equivalent
- Some models (especially smaller ones) struggle with strict JSON schema adherence
- Network timeouts or model loading issues can cause `None` returns

### Fix Applied

The application now includes robust fallback mechanisms:

1. **Detection**: Check if structured output is working during initialization
2. **Fallback**: Use manual parsing when structured output fails
3. **Error Handling**: Graceful degradation with meaningful error messages

### Technical Details

#### Before (Problematic)

```python
# This could return None
assessment_result = assessment_llm.invoke(prompt)
# Error: assessment_result.status when assessment_result is None
```

#### After (Fixed)

```python
# Test structured output capability
try:
    assessment_result = assessment_llm.invoke(prompt)
    if assessment_result is None or not hasattr(assessment_result, 'status'):
        # Fall back to manual parsing
        raw_response = regular_llm.invoke(prompt)
        assessment_result = parse_manually(raw_response.content)
except Exception:
    # Graceful error handling
    return default_response()
```

### Manual Parsing Format

The application now uses a specific response format that's easier to parse:

```
STATUS: CLEAR
REASONING: The user provided sufficient detail about their legal question
SYNTHESIZED_QUERY: Property inheritance rights under Muslim Personal Law Bangladesh
```

### Prevention Strategies

1. **Model Selection**: Use more capable models like `llama3.1` or `llama3.1:70b`
2. **Resource Allocation**: Ensure sufficient RAM (8GB+ recommended)
3. **Ollama Updates**: Keep Ollama updated to the latest version
4. **Temperature Settings**: Use temperature=0 for more consistent structured output

### Testing Your Setup

Run the test script to verify fixes:

```bash
python test_ollama_fixes.py
```

Expected output:

```
🧪 Testing Ollama Integration Fixes
==================================================

🔍 Running Ollama Connection test...
✅ Ollama connection test passed: Working

🔍 Running Structured Output test...
⚠️ Structured output failed: [expected for some models]
This is expected for some Ollama models - the app will use fallback parsing

🔍 Running Manual Parsing Fallback test...
✅ Manual parsing working: CLEAR - The query is specific enough

==================================================
📊 Test Results Summary:
------------------------------
Ollama Connection: ✅ PASS
Structured Output: ❌ FAIL  [Expected - fallback will be used]
Manual Parsing Fallback: ✅ PASS

Overall: 2/3 tests passed
⚠️ Some tests failed, but basic functionality should work.
The application includes fallback mechanisms for failed components.
```

### If Problems Persist

1. **Check Ollama Status**:

   ```bash
   ollama list
   ollama ps
   ```

2. **Verify Model Download**:

   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

3. **Test Basic Functionality**:

   ```bash
   ollama run llama3.1 "Hello, are you working?"
   ```

4. **Check System Resources**:

   - Available RAM (8GB+ recommended)
   - CPU usage during model loading
   - Disk space for models

5. **Alternative Models**:
   Try different models if current one has issues:
   ```bash
   ollama pull mistral
   ollama pull gemma2
   ```

### Performance Tips

- **For slower systems**: Use `llama3.1:8b` instead of full `llama3.1`
- **For better accuracy**: Use `llama3.1:70b` if you have 64GB+ RAM
- **For debugging**: Enable verbose logging in the application

### Support

If issues persist after trying these solutions:

1. Check the [Ollama GitHub Issues](https://github.com/ollama/ollama/issues)
2. Verify your hardware meets minimum requirements
3. Consider using the OpenAI fallback for critical applications
4. Review the application logs for specific error messages

---

**Note**: The fallback mechanisms ensure the application remains functional even when structured output fails. Most users won't notice any difference in functionality.
