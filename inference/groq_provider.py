from __future__ import annotations

import os
from typing import Optional, Any

from loguru import logger
from langchain_groq import ChatGroq

from inference.llm_provider_factory import BaseLLMProvider

LangChainLLM = Any

class GroqProvider(BaseLLMProvider):
    """
    Groq LLM provider implementation.
    """

    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("GROQ_API_KEY")

    @property
    def provider_name(self) -> str:
        return "groq"

    def validate(self) -> None:
        if not self._api_key:
            raise ValueError(
                "Missing Groq API key. "
                "Set GROQ_API_KEY in your .env file."
            )

    def build_chat_model(
        self,
        model: Optional[str],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LangChainLLM:
        """
        Build and return a ChatGroq model.
        """
        model = model or self.DEFAULT_MODEL

        logger.debug(
            f"[LLM] Groq | model={model} | temp={temperature} | max_tokens={max_tokens}"
        )

        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=self._api_key,
            **kwargs,
        )
