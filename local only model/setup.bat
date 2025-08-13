@echo off
REM Setup script for Bangladeshi Legal Assistant - Local LLM Version (Windows)
REM This script helps set up Ollama and required models on Windows

echo 🚀 Setting up Bangladeshi Legal Assistant - Local LLM Version
echo =============================================================

REM Check if Ollama is installed
where ollama >nul 2>nul
if %errorlevel% equ 0 (
    echo ✅ Ollama is already installed
) else (
    echo ❌ Ollama is not installed
    echo Please install Ollama from: https://ollama.ai/
    echo Or use: winget install ollama
    echo After installation, run this script again.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app.py" (
    echo ❌ app.py not found. Please run this script from the project directory.
    pause
    exit /b 1
)

echo Starting Ollama service...
start /B ollama serve
timeout /t 5 /nobreak >nul

echo Pulling required Ollama models...
echo This may take several minutes depending on your internet connection...

echo Pulling llama3.1 (recommended model)...
ollama pull llama3.1
if %errorlevel% neq 0 (
    echo ❌ Failed to pull llama3.1 model
    pause
    exit /b 1
)

echo Pulling nomic-embed-text (embedding model)...
ollama pull nomic-embed-text
if %errorlevel% neq 0 (
    echo ❌ Failed to pull nomic-embed-text model
    pause
    exit /b 1
)

set /p additional="Do you want to pull additional models? (y/n): "
if /i "%additional%"=="y" (
    echo Pulling llama3.1:8b (faster, smaller model)...
    ollama pull llama3.1:8b
    
    echo Pulling mistral (alternative model)...
    ollama pull mistral
)

echo Installing Python dependencies...
if not exist "requirements.txt" (
    echo ❌ requirements.txt not found
    pause
    exit /b 1
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install Python dependencies
    echo Make sure you have Python and pip installed
    pause
    exit /b 1
)

echo Creating .env template...
if not exist ".env" (
    echo # Optional: Only if using OpenAI embeddings > .env
    echo # OPENAI_API_KEY=your_openai_api_key_here >> .env
    echo. >> .env
    echo # Required: For web search fallback >> .env
    echo TAVILY_API_KEY=your_tavily_api_key_here >> .env
    echo ✅ Created .env template
    echo Please edit .env file and add your API keys
) else (
    echo ⚠️  .env file already exists
)

echo Verifying setup...

REM Check models
ollama list | findstr "llama3.1" >nul
if %errorlevel% equ 0 (
    echo ✅ llama3.1 model available
) else (
    echo ❌ llama3.1 model not found
    pause
    exit /b 1
)

ollama list | findstr "nomic-embed-text" >nul
if %errorlevel% equ 0 (
    echo ✅ nomic-embed-text model available
) else (
    echo ❌ nomic-embed-text model not found
    pause
    exit /b 1
)

REM Check Python dependencies
python -c "import streamlit, langchain_ollama, ollama" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Required Python packages available
) else (
    echo ❌ Some Python packages are missing
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo =============================================
echo Next steps:
echo 1. Edit .env file and add your API keys
echo 2. Start the application: streamlit run app.py
echo.
echo The application will automatically start Ollama if needed.
echo.
pause
