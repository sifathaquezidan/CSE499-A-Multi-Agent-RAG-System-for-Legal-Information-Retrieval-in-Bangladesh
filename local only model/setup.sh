#!/bin/bash

# Setup script for Bangladeshi Legal Assistant - Local LLM Version
# This script helps set up Ollama and required models

echo "🚀 Setting up Bangladeshi Legal Assistant - Local LLM Version"
echo "============================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Ollama is installed
check_ollama() {
    echo -e "${YELLOW}Checking Ollama installation...${NC}"
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✅ Ollama is installed${NC}"
        return 0
    else
        echo -e "${RED}❌ Ollama is not installed${NC}"
        return 1
    fi
}

# Install Ollama
install_ollama() {
    echo -e "${YELLOW}Installing Ollama...${NC}"
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo -e "${RED}Please install Homebrew or download Ollama manually from https://ollama.ai/${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        echo -e "${YELLOW}For Windows, please download and install Ollama from: https://ollama.ai/${NC}"
        echo -e "${YELLOW}Or use: winget install ollama${NC}"
        exit 1
    else
        echo -e "${RED}Unsupported OS. Please install Ollama manually from https://ollama.ai/${NC}"
        exit 1
    fi
}

# Pull required models
pull_models() {
    echo -e "${YELLOW}Pulling required Ollama models...${NC}"
    
    # Start Ollama service if not running
    echo -e "${YELLOW}Starting Ollama service...${NC}"
    ollama serve &
    OLLAMA_PID=$!
    sleep 5
    
    # Core models
    echo -e "${YELLOW}Pulling llama3.1 (recommended model)...${NC}"
    ollama pull llama3.1
    
    echo -e "${YELLOW}Pulling nomic-embed-text (embedding model)...${NC}"
    ollama pull nomic-embed-text
    
    # Optional: Offer to pull additional models
    echo -e "${YELLOW}Do you want to pull additional models? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${YELLOW}Pulling llama3.1:8b (faster, smaller model)...${NC}"
        ollama pull llama3.1:8b
        
        echo -e "${YELLOW}Pulling mistral (alternative model)...${NC}"
        ollama pull mistral
    fi
    
    # Stop Ollama service started by script
    kill $OLLAMA_PID 2>/dev/null
    
    echo -e "${GREEN}✅ Models downloaded successfully${NC}"
}

# Install Python dependencies
install_python_deps() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        echo -e "${RED}❌ requirements.txt not found. Make sure you're in the project directory.${NC}"
        exit 1
    fi
    
    # Install dependencies
    pip install -r requirements.txt
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Python dependencies installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install Python dependencies${NC}"
        exit 1
    fi
}

# Create environment file template
create_env_template() {
    echo -e "${YELLOW}Creating .env template...${NC}"
    
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# Optional: Only if using OpenAI embeddings
# OPENAI_API_KEY=your_openai_api_key_here

# Required: For web search fallback
TAVILY_API_KEY=your_tavily_api_key_here
EOF
        echo -e "${GREEN}✅ Created .env template${NC}"
        echo -e "${YELLOW}Please edit .env file and add your API keys${NC}"
    else
        echo -e "${YELLOW}⚠️  .env file already exists${NC}"
    fi
}

# Verify setup
verify_setup() {
    echo -e "${YELLOW}Verifying setup...${NC}"
    
    # Check Ollama
    if ! check_ollama; then
        echo -e "${RED}❌ Ollama verification failed${NC}"
        return 1
    fi
    
    # Check if models are available
    echo -e "${YELLOW}Checking available models...${NC}"
    if ollama list | grep -q "llama3.1"; then
        echo -e "${GREEN}✅ llama3.1 model available${NC}"
    else
        echo -e "${RED}❌ llama3.1 model not found${NC}"
        return 1
    fi
    
    if ollama list | grep -q "nomic-embed-text"; then
        echo -e "${GREEN}✅ nomic-embed-text model available${NC}"
    else
        echo -e "${RED}❌ nomic-embed-text model not found${NC}"
        return 1
    fi
    
    # Check Python dependencies
    echo -e "${YELLOW}Checking Python dependencies...${NC}"
    if python -c "import streamlit, langchain_ollama, ollama" 2>/dev/null; then
        echo -e "${GREEN}✅ Required Python packages available${NC}"
    else
        echo -e "${RED}❌ Some Python packages are missing${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Setup verification complete!${NC}"
    return 0
}

# Main execution
main() {
    echo -e "${YELLOW}Starting setup process...${NC}"
    
    # Check if we're in the right directory
    if [[ ! -f "app.py" ]]; then
        echo -e "${RED}❌ app.py not found. Please run this script from the project directory.${NC}"
        exit 1
    fi
    
    # Install Ollama if needed
    if ! check_ollama; then
        install_ollama
    fi
    
    # Pull models
    pull_models
    
    # Install Python dependencies
    install_python_deps
    
    # Create environment template
    create_env_template
    
    # Verify setup
    if verify_setup; then
        echo -e "${GREEN}"
        echo "🎉 Setup completed successfully!"
        echo "============================================="
        echo "Next steps:"
        echo "1. Edit .env file and add your API keys"
        echo "2. Ensure Ollama is running: ollama serve"
        echo "3. Start the application: streamlit run app.py"
        echo -e "${NC}"
    else
        echo -e "${RED}❌ Setup verification failed. Please check the errors above.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"
