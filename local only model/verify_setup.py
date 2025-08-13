#!/usr/bin/env python3
"""
Verification script for Bangladeshi Legal Assistant - Local LLM Version
This script checks if all components are properly installed and configured.
"""

import os
import sys
import subprocess
import importlib
from typing import Dict, List, Tuple

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_status(message: str, status: str, details: str = ""):
    """Print status with color coding"""
    status_symbols = {
        "PASS": "✅",
        "FAIL": "❌", 
        "WARN": "⚠️",
        "INFO": "ℹ️"
    }
    symbol = status_symbols.get(status, "?")
    print(f"{symbol} {message}")
    if details:
        print(f"   {details}")

def check_python_version() -> bool:
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", "PASS")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", "FAIL", 
                    "Python 3.8+ required")
        return False

def check_package_installation() -> Tuple[bool, List[str]]:
    """Check if required Python packages are installed"""
    required_packages = [
        "streamlit",
        "langchain",
        "langchain_ollama", 
        "langchain_openai",
        "langchain_community",
        "langchain_chroma",
        "langgraph",
        "chromadb",
        "ollama",
        "pydantic",
        "python_dotenv",
        "PyMuPDF",
        "tavily_python"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle packages with different import names
            if package == "python_dotenv":
                importlib.import_module("dotenv")
            elif package == "PyMuPDF":
                importlib.import_module("fitz")
            elif package == "tavily_python":
                importlib.import_module("tavily")
            else:
                importlib.import_module(package)
            print_status(f"Package: {package}", "PASS")
        except ImportError:
            print_status(f"Package: {package}", "FAIL", "Not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_ollama_installation() -> bool:
    """Check if Ollama is installed and running"""
    try:
        # Check if ollama command exists
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_status("Ollama installation", "PASS", f"Version: {result.stdout.strip()}")
            return True
        else:
            print_status("Ollama installation", "FAIL", "Command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("Ollama installation", "FAIL", "Ollama not found in PATH")
        return False

def check_ollama_service() -> bool:
    """Check if Ollama service is running"""
    try:
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            print_status("Ollama service", "PASS", "Service is running")
            return True
        else:
            print_status("Ollama service", "FAIL", "Service not responding")
            return False
    except subprocess.TimeoutExpired:
        print_status("Ollama service", "FAIL", "Service timeout")
        return False

def check_ollama_models() -> Tuple[bool, List[str]]:
    """Check if required Ollama models are available"""
    required_models = ["llama3.1", "nomic-embed-text"]
    
    try:
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            print_status("Ollama models check", "FAIL", "Cannot list models")
            return False, required_models
        
        available_models = result.stdout
        missing_models = []
        
        for model in required_models:
            if model in available_models:
                print_status(f"Model: {model}", "PASS")
            else:
                print_status(f"Model: {model}", "FAIL", "Not downloaded")
                missing_models.append(model)
        
        return len(missing_models) == 0, missing_models
        
    except subprocess.TimeoutExpired:
        print_status("Ollama models check", "FAIL", "Timeout checking models")
        return False, required_models

def check_project_structure() -> bool:
    """Check if project files and directories exist"""
    required_files = ["app.py", "requirements.txt"]
    required_dirs = ["data"]
    
    all_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print_status(f"File: {file}", "PASS")
        else:
            print_status(f"File: {file}", "FAIL", "File not found")
            all_exist = False
    
    for dir in required_dirs:
        if os.path.exists(dir) and os.path.isdir(dir):
            # Check if data directory has PDF files
            if dir == "data":
                pdf_files = [f for f in os.listdir(dir) if f.lower().endswith('.pdf')]
                if pdf_files:
                    print_status(f"Directory: {dir}", "PASS", f"Contains {len(pdf_files)} PDF files")
                else:
                    print_status(f"Directory: {dir}", "WARN", "No PDF files found")
            else:
                print_status(f"Directory: {dir}", "PASS")
        else:
            print_status(f"Directory: {dir}", "FAIL", "Directory not found")
            all_exist = False
    
    return all_exist

def check_environment_variables() -> Dict[str, bool]:
    """Check environment variables"""
    env_status = {}
    
    # Check .env file
    if os.path.exists(".env"):
        print_status(".env file", "PASS", "Configuration file exists")
        
        # Load and check variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Check Tavily API key (required)
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                print_status("TAVILY_API_KEY", "PASS", "Found in environment")
                env_status["tavily"] = True
            else:
                print_status("TAVILY_API_KEY", "WARN", "Not set (required for web search)")
                env_status["tavily"] = False
            
            # Check OpenAI API key (optional)
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                print_status("OPENAI_API_KEY", "PASS", "Found in environment (optional)")
                env_status["openai"] = True
            else:
                print_status("OPENAI_API_KEY", "INFO", "Not set (optional for embeddings)")
                env_status["openai"] = False
                
        except Exception as e:
            print_status("Environment loading", "FAIL", f"Error: {e}")
            env_status["tavily"] = False
            env_status["openai"] = False
    else:
        print_status(".env file", "WARN", "No .env file found - create one for API keys")
        env_status["tavily"] = False
        env_status["openai"] = False
    
    return env_status

def run_basic_functionality_test() -> bool:
    """Run a basic test to see if core components can be imported"""
    try:
        print_status("Testing core imports", "INFO", "This may take a moment...")
        
        # Test Ollama connection
        import ollama
        try:
            client = ollama.Client()
            models = client.list()
            print_status("Ollama client connection", "PASS")
        except Exception as e:
            print_status("Ollama client connection", "FAIL", f"Error: {e}")
            return False
        
        # Test LangChain Ollama
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        print_status("LangChain Ollama imports", "PASS")
        
        # Test Streamlit
        import streamlit
        print_status("Streamlit import", "PASS")
        
        # Test ChromaDB
        import chromadb
        print_status("ChromaDB import", "PASS")
        
        return True
        
    except Exception as e:
        print_status("Basic functionality test", "FAIL", f"Error: {e}")
        return False

def generate_recommendations(results: Dict):
    """Generate recommendations based on test results"""
    print_header("RECOMMENDATIONS")
    
    recommendations = []
    
    if not results.get("python_version", False):
        recommendations.append("⚠️  Upgrade to Python 3.8 or higher")
    
    if not results.get("packages_installed", False):
        missing = results.get("missing_packages", [])
        recommendations.append(f"📦 Install missing packages: pip install {' '.join(missing)}")
    
    if not results.get("ollama_installed", False):
        recommendations.append("🔧 Install Ollama from https://ollama.ai/")
    
    if not results.get("ollama_service", False):
        recommendations.append("🚀 Start Ollama service: ollama serve")
    
    if not results.get("models_available", False):
        missing = results.get("missing_models", [])
        for model in missing:
            recommendations.append(f"🤖 Download model: ollama pull {model}")
    
    if not results.get("project_structure", False):
        recommendations.append("📁 Ensure you're in the correct project directory")
        recommendations.append("📄 Add PDF files to the 'data' directory")
    
    if not results.get("env_vars", {}).get("tavily", False):
        recommendations.append("🔑 Add TAVILY_API_KEY to .env file")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("🎉 All checks passed! You're ready to run the application.")
        print("\nTo start the application:")
        print("   streamlit run app.py")

def main():
    """Main verification function"""
    print_header("BANGLADESHI LEGAL ASSISTANT - SYSTEM VERIFICATION")
    
    results = {}
    
    # Check Python version
    print_header("Python Environment")
    results["python_version"] = check_python_version()
    
    # Check package installation
    print_header("Python Packages")
    packages_ok, missing_packages = check_package_installation()
    results["packages_installed"] = packages_ok
    results["missing_packages"] = missing_packages
    
    # Check Ollama installation
    print_header("Ollama Installation")
    results["ollama_installed"] = check_ollama_installation()
    results["ollama_service"] = check_ollama_service()
    
    # Check Ollama models
    print_header("Ollama Models")
    models_ok, missing_models = check_ollama_models()
    results["models_available"] = models_ok
    results["missing_models"] = missing_models
    
    # Check project structure
    print_header("Project Structure")
    results["project_structure"] = check_project_structure()
    
    # Check environment variables
    print_header("Environment Configuration")
    results["env_vars"] = check_environment_variables()
    
    # Run basic functionality test
    print_header("Functionality Test")
    results["functionality"] = run_basic_functionality_test()
    
    # Generate recommendations
    generate_recommendations(results)
    
    # Overall status
    print_header("OVERALL STATUS")
    critical_checks = [
        results["python_version"],
        results["packages_installed"], 
        results["ollama_installed"],
        results["ollama_service"],
        results["models_available"],
        results["project_structure"]
    ]
    
    if all(critical_checks):
        print_status("System Status", "PASS", "Ready to run the application!")
        return 0
    else:
        print_status("System Status", "FAIL", "Please address the issues above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
