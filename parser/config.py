"""
PDFlex Configuration Module
Centralizes all global application settings.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from functools import lru_cache

load_dotenv()


class DocumentType(str, Enum):
    """Supported PDF document types."""
    SIMPLE_TEXT = "Simple Text"
    SCANNED_IMAGE = "Scanned/Image"
    COMPLEX_TABLES = "Complex Tables"
    SCIENTIFIC = "Scientific/Formulas"
    UNKNOWN = "Unknown"


class LLMConfig(BaseModel):
    """Language model configuration."""
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    api_key: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=2048)

class PathConfig(BaseModel):
    """Filesystem paths configuration."""
    output_dir: Path = Field(default=Path("output"))
    log_dir: Path = Field(default=Path("logs"))
    temp_dir: Path = Field(default=Path("/tmp/pdflex"))


class AppConfig(BaseModel):
    """Global configuration for the PDFlex application."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    debug: bool = Field(default=False)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            ),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


_default_config: Optional[AppConfig] = None


@lru_cache
def get_config() -> AppConfig:
    return AppConfig.from_env()

def set_config(config: AppConfig) -> None:
    """Override the default configuration (useful for testing)."""
    global _default_config
    _default_config = config
