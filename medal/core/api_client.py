"""
API client abstractions for different LLM providers.
"""
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from openai import AzureOpenAI

from medal.config import APIConfig


@dataclass
class APIRequest:
    """Represents an API request."""
    custom_id: str
    method: str = "POST"
    url: str = "/v1/chat/completions"
    body: Dict[str, Any] = None


class BaseAPIClient(ABC):
    """Base class for API clients."""
    
    def __init__(self, config: APIConfig):
        self.config = config
    
    @abstractmethod
    def get_client(self):
        """Get the API client instance."""
        pass
    
    @abstractmethod
    def get_url(self) -> str:
        """Get the API endpoint URL."""
        pass
    
    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        pass


class OpenAIClient(BaseAPIClient):
    """OpenAI API client."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self._client = None
    
    def get_client(self):
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.config.api_key)
        return self._client
    
    def get_url(self) -> str:
        return "https://api.openai.com/v1/chat/completions"
    
    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}


class AzureOpenAIClient(BaseAPIClient):
    """Azure OpenAI API client."""
    
    def __init__(self, config: APIConfig, deployment_url: str):
        super().__init__(config)
        self.deployment_url = deployment_url
        self._client = None
    
    def get_client(self):
        if self._client is None:
            self._client = AzureOpenAI(
                api_key=self.config.api_key,
                api_version="2025-01-01-preview",
                azure_endpoint=self.deployment_url.split("/openai")[0]
            )
        return self._client
    
    def get_url(self) -> str:
        return self.deployment_url
    
    def get_headers(self) -> Dict[str, str]:
        return {"api-key": self.config.api_key}


class DeepSeekClient(BaseAPIClient):
    """DeepSeek API client."""
    
    def get_client(self):
        return None  # Not using OpenAI SDK
    
    def get_url(self) -> str:
        return "https://api.deepseek.com/v1/chat/completions"
    
    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}


class OpenRouterClient(BaseAPIClient):
    """OpenRouter API client."""
    
    def get_client(self):
        return None  # Not using OpenAI SDK
    
    def get_url(self) -> str:
        return "https://openrouter.ai/api/v1/chat/completions"
    
    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}


class GoogleClient(BaseAPIClient):
    """Google Gemini API client."""
    
    def get_client(self):
        return None  # Not using OpenAI SDK
    
    def get_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    
    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}


def create_api_client(provider: str, config: Optional[APIConfig] = None, **kwargs) -> BaseAPIClient:
    """
    Factory function to create API clients.
    
    Args:
        provider: Provider name ('openai', 'azure', 'deepseek', 'openrouter', 'google')
        config: API configuration
        **kwargs: Additional provider-specific arguments
    
    Returns:
        API client instance
    """
    if config is None:
        config = APIConfig(provider=provider)
    
    provider_map = {
        "openai": OpenAIClient,
        "azure": lambda c: AzureOpenAIClient(c, kwargs.get("deployment_url", "")),
        "deepseek": DeepSeekClient,
        "openrouter": OpenRouterClient,
        "google": GoogleClient,
    }
    
    client_class = provider_map.get(provider.lower())
    if client_class is None:
        raise ValueError(f"Unknown provider: {provider}")
    
    return client_class(config)
