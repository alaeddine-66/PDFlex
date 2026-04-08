"""
PDFlex - Base Extractor Interfaces
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class BaseExtractor(ABC):
    """
    Defines the standard interface for all document extractors.
    """

    @abstractmethod
    def extract(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """
        Extracts text from the document.
        Returns a tuple: (extracted_text, metadata).
        """
        ...
