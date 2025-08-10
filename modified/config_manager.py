"""
Configuration management for the Legal RAG application
Handles LLM providers, API keys, and other settings with Streamlit caching
"""

import streamlit as st
import os
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv

from llm_providers import (
    LLMConfig, LLMManager, LLMProvider, 
    create_llm_config_ui, create_embedding_config_ui,
    OpenAIEmbeddingProvider
)

load_dotenv()


class AppConfig:
    """Application configuration manager"""
    
    def __init__(self):
        self.llm_config: Optional[LLMConfig] = None
        self.llm_manager: Optional[LLMManager] = None
        self.embedding_model: Optional[Any] = None
        self.tavily_api_key: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if configuration is valid and complete"""
        return (
            self.llm_config is not None and
            self.llm_manager is not None and
            self.embedding_model is not None and
            self.tavily_api_key is not None
        )


@st.cache_resource
def get_cached_llm_manager(config_dict: Dict[str, Any]) -> Optional[LLMManager]:
    """Create and cache LLM manager based on configuration"""
    try:
        config = LLMConfig(**config_dict)
        manager = LLMManager(config)
        
        # Test the configuration by creating a model
        test_model = manager.get_chat_model()
        if test_model is None:
            return None
            
        return manager
    except Exception as e:
        st.error(f"Failed to initialize LLM manager: {e}")
        return None


@st.cache_resource
def get_cached_embedding_model(provider: str, api_key: str, model: str) -> Optional[Any]:
    """Create and cache embedding model"""
    try:
        if provider == "OpenAI":
            embedding_provider = OpenAIEmbeddingProvider(api_key, model)
            return embedding_provider.get_embedding_model()
    except Exception as e:
        st.error(f"Failed to initialize embedding model: {e}")
        return None


def create_configuration_ui() -> AppConfig:
    """Create the configuration UI and return AppConfig"""
    config = AppConfig()
    
    st.sidebar.header("🛠️ Configuration")
    
    # Create tabs for better organization
    config_tab1, config_tab2 = st.sidebar.tabs(["LLM Settings", "Other APIs"])
    
    with config_tab1:
        # LLM Configuration
        llm_config = create_llm_config_ui()
        
        if llm_config:
            # Store API keys in session state for persistence
            if llm_config.provider == LLMProvider.OPENAI and llm_config.api_key:
                st.session_state["openai_api_key"] = llm_config.api_key
            elif llm_config.provider == LLMProvider.GEMINI and llm_config.api_key:
                st.session_state["gemini_api_key"] = llm_config.api_key
            
            # Create cached LLM manager
            config_dict = llm_config.dict()
            llm_manager = get_cached_llm_manager(config_dict)
            
            if llm_manager:
                config.llm_config = llm_config
                config.llm_manager = llm_manager
                st.success(f"✅ {llm_config.provider.value.title()} LLM ready: {llm_config.model}")
        
        st.markdown("---")
        
        # Embedding Configuration
        embedding_model = create_embedding_config_ui()
        if embedding_model:
            config.embedding_model = embedding_model
    
    with config_tab2:
        # Tavily API Key
        st.subheader("🔍 Web Search (Tavily)")
        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",
            value=os.getenv("TAVILY_API_KEY", ""),
            help="Required for web search functionality",
            key="tavily_api_key_input"
        )
        
        if tavily_key:
            config.tavily_api_key = tavily_key
            st.success("✅ Tavily API key configured")
        else:
            st.warning("Tavily API key required for web search")
    
    return config


def get_legacy_config() -> Tuple[Optional[str], Optional[str]]:
    """Get legacy OpenAI and Tavily API keys for backward compatibility"""
    openai_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        value=os.getenv("OPENAI_API_KEY", ""),
        key="legacy_openai_key"
    )
    
    tavily_key = st.text_input(
        "Tavily API Key", 
        type="password", 
        value=os.getenv("TAVILY_API_KEY", ""),
        key="legacy_tavily_key"
    )
    
    return openai_key, tavily_key


def create_legacy_compatibility_wrapper(openai_api_key: str, tavily_api_key: str) -> AppConfig:
    """Create AppConfig from legacy API keys for backward compatibility"""
    config = AppConfig()
    
    # Create default OpenAI configuration
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=openai_api_key
    )
    
    # Create LLM manager
    config_dict = llm_config.dict()
    llm_manager = get_cached_llm_manager(config_dict)
    
    if llm_manager:
        config.llm_config = llm_config
        config.llm_manager = llm_manager
    
    # Create embedding model
    embedding_model = get_cached_embedding_model("OpenAI", openai_api_key, "text-embedding-3-large")
    if embedding_model:
        config.embedding_model = embedding_model
    
    # Set Tavily key
    config.tavily_api_key = tavily_api_key
    
    return config


def show_configuration_status(config: AppConfig):
    """Display configuration status in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status")
    
    status_items = []
    
    if config.llm_config and config.llm_manager:
        status_items.append(f"✅ LLM: {config.llm_config.provider.value.title()} ({config.llm_config.model})")
    else:
        status_items.append("❌ LLM: Not configured")
    
    if config.embedding_model:
        status_items.append("✅ Embeddings: Ready")
    else:
        status_items.append("❌ Embeddings: Not configured")
    
    if config.tavily_api_key:
        status_items.append("✅ Web Search: Ready")
    else:
        status_items.append("❌ Web Search: Not configured")
    
    for item in status_items:
        st.sidebar.write(item)
    
    if config.is_valid():
        st.sidebar.success("🎉 All components ready!")
    else:
        st.sidebar.error("⚠️ Configuration incomplete")


def get_provider_info_text(provider: LLMProvider) -> str:
    """Get informational text about each provider"""
    info = {
        LLMProvider.OPENAI: """
        **OpenAI (Recommended)**
        - Most reliable and well-tested
        - Best performance for legal tasks
        - Supports structured output
        - Requires API key and credits
        """,
        LLMProvider.OLLAMA: """
        **Ollama (Local)**
        - Runs locally on your machine
        - No API costs or usage limits
        - Privacy-focused (data stays local)
        - Requires Ollama installation
        - May have slower performance
        """,
        LLMProvider.GEMINI: """
        **Google Gemini**
        - Competitive performance
        - Good for complex reasoning
        - Requires Google API key
        - Supports structured output
        """
    }
    return info.get(provider, "")


def show_provider_help():
    """Show help information about different providers"""
    with st.sidebar.expander("🤔 Which Provider to Choose?", expanded=False):
        st.markdown("""
        **For most users: OpenAI** is recommended for the best experience.
        
        **For privacy/cost concerns: Ollama** runs locally.
        
        **For experimentation: Gemini** offers competitive capabilities.
        """)
        
        for provider in LLMProvider:
            st.markdown(get_provider_info_text(provider))
