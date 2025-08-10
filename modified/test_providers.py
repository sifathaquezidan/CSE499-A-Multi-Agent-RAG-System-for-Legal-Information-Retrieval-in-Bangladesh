"""
Test script for the multi-LLM provider system
"""

import os
import sys
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(__file__))

load_dotenv()

def test_llm_providers():
    """Test different LLM providers"""
    from llm_providers import LLMConfig, LLMManager, LLMProvider
    
    print("=== Testing LLM Providers ===\n")
    
    # Test OpenAI
    print("1. Testing OpenAI Provider:")
    try:
        openai_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        openai_manager = LLMManager(openai_config)
        if openai_manager.validate_config():
            print("   ✅ OpenAI configuration valid")
            
            # Test basic chat
            chat_model = openai_manager.get_chat_model()
            response = chat_model.invoke("What is the capital of Bangladesh?")
            print(f"   ✅ OpenAI response: {response.content[:100]}...")
            
            # Test structured output
            from pydantic import BaseModel, Field
            class TestModel(BaseModel):
                country: str = Field(description="The country name")
                capital: str = Field(description="The capital city")
            
            structured_model = openai_manager.get_structured_output_model(TestModel)
            structured_response = structured_model.invoke("What is the capital of Bangladesh?")
            print(f"   ✅ Structured output: {structured_response}")
            
        else:
            print("   ❌ OpenAI configuration invalid")
    except Exception as e:
        print(f"   ❌ OpenAI test failed: {e}")
    
    print()
    
    # Test Ollama (if available)
    print("2. Testing Ollama Provider:")
    try:
        ollama_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama2",
            temperature=0.0,
            base_url="http://localhost:11434"
        )
        
        ollama_manager = LLMManager(ollama_config)
        if ollama_manager.validate_config():
            print("   ✅ Ollama configuration valid")
            
            # Test basic chat (this might fail if Ollama isn't running)
            try:
                chat_model = ollama_manager.get_chat_model()
                response = chat_model.invoke("What is 2+2?")
                print(f"   ✅ Ollama response: {response.content[:100]}...")
            except Exception as e:
                print(f"   ⚠️ Ollama test failed (is Ollama running?): {e}")
        else:
            print("   ❌ Ollama configuration invalid")
    except Exception as e:
        print(f"   ❌ Ollama test failed: {e}")
    
    print()
    
    # Test Gemini
    print("3. Testing Gemini Provider:")
    try:
        gemini_config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model="gemini-pro",
            temperature=0.0,
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        gemini_manager = LLMManager(gemini_config)
        if gemini_manager.validate_config():
            print("   ✅ Gemini configuration valid")
            
            # Test basic chat
            try:
                chat_model = gemini_manager.get_chat_model()
                response = chat_model.invoke("What is the capital of Bangladesh?")
                print(f"   ✅ Gemini response: {response.content[:100]}...")
            except Exception as e:
                print(f"   ⚠️ Gemini test failed: {e}")
        else:
            print("   ❌ Gemini configuration invalid (check API key)")
    except Exception as e:
        print(f"   ❌ Gemini test failed: {e}")

def test_config_manager():
    """Test the configuration manager"""
    print("\n=== Testing Configuration Manager ===\n")
    
    try:
        from config_manager import AppConfig, create_legacy_compatibility_wrapper
        
        # Test legacy compatibility
        openai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        if openai_key and tavily_key:
            print("Testing legacy compatibility wrapper:")
            app_config = create_legacy_compatibility_wrapper(openai_key, tavily_key)
            
            if app_config.is_valid():
                print("   ✅ Legacy compatibility wrapper works")
                print(f"   ✅ LLM Provider: {app_config.llm_config.provider.value}")
                print(f"   ✅ Model: {app_config.llm_config.model}")
                print(f"   ✅ Embedding model: Available")
                print(f"   ✅ Tavily key: Available")
            else:
                print("   ❌ Legacy compatibility wrapper failed")
        else:
            print("   ⚠️ Missing API keys for legacy test")
            
    except Exception as e:
        print(f"   ❌ Configuration manager test failed: {e}")

def test_embeddings():
    """Test embedding models"""
    print("\n=== Testing Embedding Models ===\n")
    
    try:
        from llm_providers import OpenAIEmbeddingProvider
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print("Testing OpenAI embeddings:")
            embedding_provider = OpenAIEmbeddingProvider(openai_key)
            embedding_model = embedding_provider.get_embedding_model()
            
            # Test embedding
            test_text = "Bangladesh is a country in South Asia."
            embeddings = embedding_model.embed_query(test_text)
            print(f"   ✅ Generated embedding of length: {len(embeddings)}")
        else:
            print("   ⚠️ Missing OpenAI API key for embedding test")
            
    except Exception as e:
        print(f"   ❌ Embedding test failed: {e}")

if __name__ == "__main__":
    print("Multi-LLM Provider System Test\n")
    print("=" * 50)
    
    test_llm_providers()
    test_config_manager()
    test_embeddings()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    
    # Print environment variables status
    print("\nEnvironment Variables Status:")
    print(f"OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
    print(f"TAVILY_API_KEY: {'✅ Set' if os.getenv('TAVILY_API_KEY') else '❌ Not set'}")
    print(f"GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Not set'}")
