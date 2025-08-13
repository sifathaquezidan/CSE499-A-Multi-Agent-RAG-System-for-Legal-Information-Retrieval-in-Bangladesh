#!/usr/bin/env python3
"""
Quick test script for the Ollama integration fixes
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ollama_connection():
    """Test basic Ollama connection"""
    try:
        from langchain_ollama import ChatOllama
        
        print("Testing Ollama connection...")
        llm = ChatOllama(model="llama3.1", temperature=0)
        
        # Simple test
        response = llm.invoke("Hello, respond with just 'Working' if you can see this.")
        print(f"✅ Ollama connection test passed: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection test failed: {e}")
        return False

def test_structured_output():
    """Test structured output with fallback"""
    try:
        from langchain_ollama import ChatOllama
        from pydantic import BaseModel, Field
        
        print("Testing structured output...")
        
        class TestOutput(BaseModel):
            status: str = Field(description="Test status")
            message: str = Field(description="Test message")
        
        llm = ChatOllama(model="llama3.1", temperature=0)
        
        try:
            structured_llm = llm.with_structured_output(TestOutput)
            result = structured_llm.invoke("Respond with status 'success' and message 'test'")
            
            if result and hasattr(result, 'status'):
                print(f"✅ Structured output working: {result.status} - {result.message}")
                return True
            else:
                print("⚠️ Structured output returned None or invalid format")
                return False
                
        except Exception as e:
            print(f"⚠️ Structured output failed: {e}")
            print("This is expected for some Ollama models - the app will use fallback parsing")
            return False
            
    except Exception as e:
        print(f"❌ Structured output test failed: {e}")
        return False

def test_manual_parsing():
    """Test manual parsing fallback"""
    try:
        print("Testing manual parsing fallback...")
        
        # Simulate a raw response
        raw_response = """
        STATUS: CLEAR
        REASONING: The query is specific enough
        SYNTHESIZED_QUERY: Test query about legal matter
        """
        
        # Import our parsing function
        from app import parse_assessment_response
        
        result = parse_assessment_response(raw_response, "Original query")
        
        if result and hasattr(result, 'status'):
            print(f"✅ Manual parsing working: {result.status} - {result.reasoning}")
            return True
        else:
            print("❌ Manual parsing failed")
            return False
            
    except Exception as e:
        print(f"❌ Manual parsing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Ollama Integration Fixes")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Structured Output", test_structured_output),  
        ("Manual Parsing Fallback", test_manual_parsing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("-" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The application should work correctly.")
    elif passed > 0:
        print("⚠️ Some tests failed, but basic functionality should work.")
        print("The application includes fallback mechanisms for failed components.")
    else:
        print("💥 All tests failed. Please check your Ollama installation.")
        print("Make sure Ollama is running and has the llama3.1 model downloaded.")
        
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
