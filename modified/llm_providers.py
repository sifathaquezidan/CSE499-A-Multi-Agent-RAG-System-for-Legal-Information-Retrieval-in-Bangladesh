"""
LLM Provider Abstraction Layer
Supports OpenAI, Ollama, and Google Gemini APIs with unified interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import streamlit as st


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"


class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    provider: LLMProvider
    model: str
    temperature: float = 0.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama
    max_tokens: Optional[int] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def get_chat_model(self) -> Any:
        """Returns the chat model instance"""
        pass
    
    @abstractmethod
    def get_structured_output_model(self, pydantic_model: type) -> Any:
        """Returns a model configured for structured output"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Returns list of available models for this provider"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validates the provider configuration"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation"""
    
    def get_chat_model(self) -> Any:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            openai_api_key=self.config.api_key,
            max_tokens=self.config.max_tokens,
            **self.config.additional_params
        )
    
    def get_structured_output_model(self, pydantic_model: type) -> Any:
        base_model = self.get_chat_model()
        return base_model.with_structured_output(pydantic_model)
    
    def get_available_models(self) -> List[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
    
    def validate_config(self) -> bool:
        return bool(self.config.api_key)


class OllamaProvider(BaseLLMProvider):
    """Ollama provider implementation"""
    
    def get_chat_model(self) -> Any:
        from langchain_community.llms import Ollama
        from langchain_community.chat_models import ChatOllama
        
        try:
            return ChatOllama(
                model=self.config.model,
                temperature=self.config.temperature,
                base_url=self.config.base_url or "http://localhost:11434",
                **self.config.additional_params
            )
        except ImportError:
            # Fallback to regular Ollama if ChatOllama not available
            return Ollama(
                model=self.config.model,
                temperature=self.config.temperature,
                base_url=self.config.base_url or "http://localhost:11434",
                **self.config.additional_params
            )
    
    def get_structured_output_model(self, pydantic_model: type) -> Any:
        # Ollama doesn't natively support structured output, so we'll use the base model
        # and handle parsing manually or use output parsers
        base_model = self.get_chat_model()
        try:
            return base_model.with_structured_output(pydantic_model)
        except AttributeError:
            # If structured output not supported, return base model
            # The calling code should handle this case
            return base_model
    
    def get_available_models(self) -> List[str]:
        # These are common Ollama models - in a real implementation,
        # you might want to query the Ollama API for available models
        return [
            "llama3.2:3b",
            "llama3.1:8b",
        ]
    
    def validate_config(self) -> bool:
        # For Ollama, we just need a valid base_url (defaults to localhost)
        return True


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation"""
    
    def get_chat_model(self) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        return ChatGoogleGenerativeAI(
            model=self.config.model,
            temperature=self.config.temperature,
            google_api_key=self.config.api_key,
            max_tokens=self.config.max_tokens,
            **self.config.additional_params
        )
    
    def get_structured_output_model(self, pydantic_model: type) -> Any:
        base_model = self.get_chat_model()
        try:
            return base_model.with_structured_output(pydantic_model)
        except AttributeError:
            # If structured output not supported, return base model
            return base_model
    
    def get_available_models(self) -> List[str]:
        return [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
    
    def validate_config(self) -> bool:
        return bool(self.config.api_key)


class LLMProviderFactory:
    """Factory class for creating LLM providers"""
    
    _providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.OLLAMA: OllamaProvider,
        LLMProvider.GEMINI: GeminiProvider
    }
    
    @classmethod
    def create_provider(cls, config: LLMConfig) -> BaseLLMProvider:
        """Create a provider instance based on configuration"""
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> List[LLMProvider]:
        """Get list of available providers"""
        return list(cls._providers.keys())


class LLMManager:
    """Manages LLM providers and provides unified interface"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = LLMProviderFactory.create_provider(config)
    
    def get_chat_model(self) -> Any:
        """Get chat model from current provider"""
        return self.provider.get_chat_model()
    
    def get_structured_output_model(self, pydantic_model: type) -> Any:
        """Get structured output model from current provider"""
        return self.provider.get_structured_output_model(pydantic_model)
    
    def get_available_models(self) -> List[str]:
        """Get available models for current provider"""
        return self.provider.get_available_models()
    
    def validate_config(self) -> bool:
        """Validate current configuration"""
        return self.provider.validate_config()
    
    def update_config(self, new_config: LLMConfig):
        """Update configuration and recreate provider"""
        self.config = new_config
        self.provider = LLMProviderFactory.create_provider(new_config)


def create_llm_config_ui() -> Optional[LLMConfig]:
    """Create Streamlit UI for LLM configuration"""
    st.subheader("🤖 LLM Configuration")
    
    # Provider selection
    provider = st.selectbox(
        "Select LLM Provider",
        options=[p.value for p in LLMProvider],
        format_func=lambda x: {
            "openai": "OpenAI (GPT-4, GPT-3.5)",
            "ollama": "Ollama (Local Models)",
            "gemini": "Google Gemini"
        }.get(x, x.title()),
        key="llm_provider_select"
    )
    
    provider_enum = LLMProvider(provider)
    
    # Get available models for selected provider
    temp_config = LLMConfig(provider=provider_enum, model="dummy")
    temp_provider = LLMProviderFactory.create_provider(temp_config)
    available_models = temp_provider.get_available_models()
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        options=available_models,
        key="llm_model_select"
    )
    
    # Temperature setting
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Controls randomness. 0.0 = deterministic, 2.0 = very creative",
        key="llm_temperature"
    )
    
    # Provider-specific configurations
    api_key = None
    base_url = None
    max_tokens = None
    
    if provider_enum == LLMProvider.OPENAI:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
            key="openai_api_key_input"
        )
        max_tokens = st.number_input(
            "Max Tokens (optional)",
            min_value=1,
            max_value=4096,
            value=None,
            key="openai_max_tokens"
        )
    
    elif provider_enum == LLMProvider.OLLAMA:
        base_url = st.text_input(
            "Ollama Base URL",
            value="http://localhost:11434",
            help="URL where Ollama is running",
            key="ollama_base_url"
        )
        max_tokens = st.number_input(
            "Max Tokens (optional)",
            min_value=1,
            max_value=4096,
            value=None,
            key="ollama_max_tokens"
        )
    
    elif provider_enum == LLMProvider.GEMINI:
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=st.session_state.get("gemini_api_key", ""),
            key="gemini_api_key_input"
        )
        max_tokens = st.number_input(
            "Max Tokens (optional)",
            min_value=1,
            max_value=4096,
            value=None,
            key="gemini_max_tokens"
        )
    
    # Create configuration
    config = LLMConfig(
        provider=provider_enum,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens if max_tokens else None
    )
    
    # Validate configuration
    temp_provider = LLMProviderFactory.create_provider(config)
    is_valid = temp_provider.validate_config()
    
    if not is_valid:
        if provider_enum in [LLMProvider.OPENAI, LLMProvider.GEMINI]:
            st.error(f"Please provide a valid {provider_enum.value.title()} API key")
        return None
    
    st.success(f"✅ {provider_enum.value.title()} configuration is valid")
    return config


# Embedding provider abstraction
class EmbeddingProvider(ABC):
    """Base class for embedding providers"""
    
    @abstractmethod
    def get_embedding_model(self) -> Any:
        """Returns the embedding model instance"""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.api_key = api_key
        self.model = model
    
    def get_embedding_model(self) -> Any:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key
        )


def create_embedding_config_ui() -> Optional[Any]:
    """Create UI for embedding model configuration"""
    st.subheader("📊 Embedding Model Configuration")
    
    # For now, only OpenAI embeddings are supported
    # Can be extended to support other providers
    embedding_provider = st.selectbox(
        "Embedding Provider",
        options=["OpenAI"],
        key="embedding_provider_select"
    )
    
    if embedding_provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key (for embeddings)",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
            key="embedding_api_key_input"
        )
        
        model = st.selectbox(
            "Embedding Model",
            options=["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
            key="embedding_model_select"
        )
        
        if api_key:
            provider = OpenAIEmbeddingProvider(api_key, model)
            st.success("✅ Embedding configuration is valid")
            return provider.get_embedding_model()
        else:
            st.error("Please provide a valid OpenAI API key for embeddings")
            return None
    
    return None
