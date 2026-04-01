"""
PDFlex - Extraction Workers
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import os
import io
import base64

import fitz
import pytesseract
from PIL import Image
from langchain_core.messages import HumanMessage

from inference.llm_provider_factory import get_llm
from parser.config import get_config

class BaseExtractor(ABC):
    """
    Defines the standard interface for all document extractors.
    """

    @abstractmethod
    def extract(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """
        Extracts text from the PDF.
        Returns a tuple: (extracted_text, metadata).
        """
        ...

class BaseVisionExtractor(BaseExtractor):
    """
    Template Method Pattern: Handles the logic for PDF vs Image.
    Delegates the actual image reading to the subclass.
    """

    def _is_image(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]

    def extract(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        pages_text = []
        cfg = get_config()

        if self._is_image(file_path):
            text = self.process_single_image(file_path)
            pages_text.append(text)
        else:
            doc = fitz.open(str(file_path))
            for page in doc:
                pix = page.get_pixmap(dpi=cfg.ocr.dpi)
                img_bytes = pix.tobytes("jpeg")
                text = self.process_single_image_from_bytes(img_bytes)
                pages_text.append(text)
            doc.close()

        full_text = "\n\n".join(pages_text)
        return full_text, self.get_metadata(len(pages_text))

    @abstractmethod
    def process_single_image(self, image_path: Path) -> str:
        """To be implemented by Tesseract or Vision subclasses."""
        ...

    @abstractmethod
    def process_single_image_from_bytes(self, image_bytes: bytes) -> str:
        """To be implemented by Tesseract or Vision subclasses."""
        ...

    @abstractmethod
    def get_metadata(self, page_count: int) -> dict[str, Any]:
        """Returns metadata specific to the extractor."""
        ...

class TesseractExtractor(BaseVisionExtractor):

    def process_single_image(self, image_path: Path) -> str:
        return pytesseract.image_to_string(Image.open(image_path))

    def process_single_image_from_bytes(self, image_bytes: bytes) -> str:
        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img)

    def get_metadata(self, page_count: int) -> dict[str, Any]:
        return {"engine": "tesseract", "page_count": page_count}

class LLMVisionExtractor(BaseVisionExtractor):

    def _call_llm(self, b64_image: str) -> str:
        llm = get_llm()
        prompt = "You are a strict OCR engine. Transcribe the text from this image exactly as it appears."
        response = llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ])
        ])
        return response.content

    def process_single_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return self._call_llm(b64)

    def process_single_image_from_bytes(self, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        return self._call_llm(b64)

    def get_metadata(self, page_count: int) -> dict[str, Any]:
        return {"engine": "vision_llm", "page_count": page_count}


def get_vision_extractor() -> BaseVisionExtractor:
    """
    Dynamically instantiates the correct Vision extractor.
    """
    cfg = get_config()
    ocr_engine = cfg.ocr.engine.lower()

    if ocr_engine == "vision":
        return LLMVisionExtractor()
    return TesseractExtractor()
