"""
PDFlex - LLM Abstraction Layer
Centralizes the creation and configuration of all LLM clients.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger
from plugin.plugin_loader import PluginLoader


class BaseLLMProvider(ABC):
    """
    Abstract interface for any LLM provider.
    Each provider knows how to build its LangChain client.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g., 'openai', 'groq')."""
        ...

    @abstractmethod
    def build_chat_model(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LangChainLLM:
        """
        Builds and returns a configured LangChain ChatModel.
        """
        ...

    def validate(self) -> None:
        """
        Ensures that provider requirements are satisfied
        (API key present, package installed, etc.).
        Raises ValueError if something is missing.
        """
        ...

    def __repr__(self) -> str:
        return f"<LLMProvider: {self.provider_name}>"


class LLMProviderFactory:
    """
    Factory for LLM providers.

    Dynamic plugin mode (LLM_PROVIDER=plugin):
        Loads an external class from the filesystem using:
            LLM_PLUGIN_MODULE_PATH = /path/to/my_provider.py
            LLM_PLUGIN_CLASS_NAME  = MyProviderClass
    """

    @classmethod
    def create(cls) -> BaseLLMProvider:
        """
        Creates an LLM provider.
        """

        loader: PluginLoader[BaseLLMProvider] = PluginLoader("llm")

        if not loader.is_configured:
            raise EnvironmentError(
                "LLM_PROVIDER=plugin but plugin variables are not defined.\n"
                "Add the following to your .env:\n"
                "  LLM_PLUGIN_MODULE_PATH=/path/to/my_provider.py\n"
                "  LLM_PLUGIN_CLASS_NAME=MyProviderClass"
            )

        klass = loader.load()
        provider = klass()
        provider.validate()

        logger.info(f"[LLM] Provider loaded: {provider.provider_name}")
        
        return provider

def get_llm(
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> LangChainLLM:
    """
    Builds a ready-to-use LangChain LLM, driven by .env configuration.
    """
    provider = LLMProviderFactory.create()

    model = os.getenv("LLM_MODEL")

    return provider.build_chat_model(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
