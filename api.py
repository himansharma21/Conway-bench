"""
LLM API abstraction layer for the Conway's Game of Life benchmark.
Supports OpenRouter API with extensibility for other providers.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1000


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    content: str
    model: str
    response_time: float
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def query(self, prompt: str) -> LLMResponse:
        """Send a prompt to the LLM and return the response."""
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models for this provider."""
        pass


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODELS_URL = "https://openrouter.ai/api/v1/models"

    def __init__(self, config: LLMConfig):
        """
        Initialize the OpenRouter provider.

        Args:
            config: LLM configuration with API key and model settings
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration."""
        if not self.config.api_key:
            raise ValueError("API key is required for OpenRouter")

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def query(self, prompt: str) -> LLMResponse:
        """
        Send a prompt to the LLM via OpenRouter.

        Args:
            prompt: The prompt to send

        Returns:
            LLMResponse with the result
        """
        data = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        start_time = time.time()
        try:
            response = requests.post(
                self.API_URL,
                headers=self._get_headers(),
                json=data,
                timeout=60,
            )
            response.raise_for_status()
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            error = None
        except requests.exceptions.RequestException as e:
            content = ""
            error = str(e)
        except (KeyError, IndexError) as e:
            content = ""
            error = f"Invalid response format: {e}"

        response_time = time.time() - start_time

        return LLMResponse(
            content=content,
            model=self.config.model,
            response_time=response_time,
            error=error,
        )

    def list_models(self) -> list[str]:
        """
        List available models from OpenRouter.

        Returns:
            List of model IDs
        """
        try:
            response = requests.get(
                self.MODELS_URL,
                headers=self._get_headers(),
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except requests.exceptions.RequestException:
            return []


def load_config(config_path: str = "config.json") -> LLMConfig:
    """
    Load LLM configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        LLMConfig object with API settings
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)

    openrouter_config = config_data.get("openrouter", {})

    return LLMConfig(
        api_key=openrouter_config.get("api_key", ""),
        model=openrouter_config.get("model", "anthropic/claude-3.5-sonnet"),
        temperature=openrouter_config.get("temperature", 0.0),
        max_tokens=openrouter_config.get("max_tokens", 1000),
    )


def create_provider(config: LLMConfig, provider_type: str = "openrouter") -> LLMProvider:
    """
    Create an LLM provider instance.

    Args:
        config: LLM configuration
        provider_type: Type of provider ("openrouter")

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider type is not supported
    """
    providers = {
        "openrouter": OpenRouterProvider,
    }

    if provider_type not in providers:
        raise ValueError(f"Unknown provider: {provider_type}. Available: {list(providers.keys())}")

    return providers[provider_type](config)
