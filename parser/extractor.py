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
from unstructured.partition.pdf import partition_pdf
from markdownify import markdownify as md
from loguru import logger

from parser.layout_analyzer import UnstructuredLayoutAnalyzer


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


class UnstructuredExtractor(BaseExtractor):
    """
    Comprehensive PDF extractor utilizing Unstructured.io.
    """

    def __init__(self):
        """Initializes the layout analyzer, vision extractor, and application config."""
        self.layout_analyzer = UnstructuredLayoutAnalyzer()
        self.vision_extractor = get_vision_extractor()
        self.config = get_config()

    def partition(self, file_path: Path) -> list[Any]:
        """
        Partitions a PDF document into structural elements and applies layout sorting.

        Args:
            file_path (Path): The absolute or relative path to the PDF file.

        Returns:
            list[Any]: A list of sorted Unstructured elements representing the document.
        """
        output_dir = self.config.paths.temp_dir / "temp_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[UnstructuredExtractor] Partitioning PDF: {file_path.name}")
        elements = partition_pdf(
            filename=str(file_path),
            strategy="fast",
            extract_image_block_types=["Image", "Formula"],
            extract_image_block_output_dir=str(output_dir),
        )

        logger.info(f"[UnstructuredExtractor] {len(elements)} blocks detected. Applying layout sorting...")
        sorted_elements = self.layout_analyzer.sort_elements(elements)

        return sorted_elements

    def extract(self, elements: list[Any]) -> tuple[str, dict[str, Any]]:
        """
        Converts a list of parsed Unstructured elements into a formatted Markdown string.

        Args:
            elements (list[Any]): The list of sorted Unstructured elements.

        Returns:
            tuple[str, dict[str, Any]]: A tuple containing the final Markdown string 
                                        and a metadata dictionary.
        """
        final_markdown = []
        
        for element in elements:
            element_type = type(element).__name__

            if element_type in ["Text", "NarrativeText", "Title", "ListItem"]:
                prefix = "## " if element_type == "Title" else ""
                final_markdown.append(f"{prefix}{element.text}\n")

            elif element_type == "Table":
                html_table = getattr(element.metadata, "text_as_html", None)
                if html_table:
                    markdown_table = md(html_table)
                    final_markdown.append(f"{markdown_table}\n")
                else:
                    final_markdown.append(f"{element.text}\n")

            elif element_type in ["Image", "Figure", "Formula"]:
                image_path = getattr(element.metadata, "image_path", None)
                if image_path and Path(image_path).exists():
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size

                        # Filter out artifacts: extremely small images or thin layout lines
                        if (width < 30 and height < 30) or (width / height > 10) or (height / width > 10):
                            logger.debug(f"[UnstructuredExtractor] Ignored visual artifact (dimensions: {width}x{height})")
                            continue

                        # Process valid images through the Vision extractor
                        vision_text = self.vision_extractor.process_single_image(Path(image_path))
                        final_markdown.append(f"\n> **Image Extraction:**\n> {vision_text}\n")
                        
                    except Exception as e:
                        logger.error(f"[UnstructuredExtractor] AI Vision error on {image_path}: {e}")

        full_text = "\n".join(final_markdown)
        metadata = {
            "engine": "unstructured",
            "blocks_detected": len(elements)
        }

        return full_text, metadata
