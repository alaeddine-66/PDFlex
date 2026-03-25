"""
PDFlex - Classifier Node
Detects the PDF document type to route it to the appropriate worker.
"""
from __future__ import annotations

import json
import os 
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import hashlib

from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from pypdf import PdfReader

from parser.config import AppConfig, DocumentType, get_config
from parser.state import GraphState
from inference.llm_provider_factory import get_llm

from pydantic import BaseModel

CACHE_FILE = Path("cache/llm_cache.json")

SYSTEM_PROMPT = """You are an expert in PDF document classification.
Analyze the text sample and return EXACTLY one of the following types:
- "Simple Text"
- "Scanned/Image"
- "Complex Tables"
- "Scientific/Formulas"

Respond ONLY with JSON:
{{"type": "<type>", "confidence": <0.0-1.0>, "reason": "<short explanation>"}}"""


def load_cache() -> dict[str, tuple]:
    if CACHE_FILE.exists():
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return {k: tuple(v) for k, v in data.items()}
    return {}

def save_cache(cache: dict[str, tuple]):
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class LLMClassificationOutput(BaseModel):
    type: str
    confidence: float
    reason: str

class BaseClassifier(ABC):
    """Abstract interface for any PDF classifier."""

    @abstractmethod
    def classify(self, file_path: Path) -> tuple[DocumentType, float, dict[str, Any]]:
        """
        Returns (document_type, confidence_score, metadata).
        """
        ...


class HeuristicClassifier(BaseClassifier):
    """
    Heuristic classifier (fast, no LLM).
    Analyzes PyMuPDF metadata and a sample from the first page.
    """

    def classify(self, file_path: Path) -> tuple[DocumentType, float, dict[str, Any]]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required: pip install pymupdf")

        doc = fitz.open(str(file_path))
        meta: dict[str, Any] = {
            "page_count": doc.page_count,
            "has_images": False,
            "has_text": False,
            "has_tables": False,
            "image_ratio": 0.0,
            "text_density": 0.0,
        }

        try:
            page = doc[0]
            text = page.get_text("text")
            blocks = page.get_text("blocks")
            images = page.get_images(full=True)

            page_area = page.rect.width * page.rect.height
            image_area = sum(
                abs((b[2] - b[0]) * (b[3] - b[1]))
                for b in page.get_image_info()
            ) if images else 0

            meta["has_images"] = bool(images)
            meta["has_text"] = bool(text.strip())
            meta["image_ratio"] = image_area / page_area if page_area else 0.0
            meta["text_density"] = len(text.strip()) / max(page_area, 1)
            meta["text_sample"] = text[:500]

            doc_type, confidence = self._apply_heuristics(meta, text, blocks)
        finally:
            doc.close()

        return doc_type, confidence, meta

    def _apply_heuristics(
        self,
        meta: dict,
        text: str,
        blocks: list,
    ) -> tuple[DocumentType, float]:
        """Apply heuristic rules in priority order."""

        # 1. Scanned: many images, little extractable text
        if meta["image_ratio"] > 0.5 and not meta["has_text"]:
            return DocumentType.SCANNED_IMAGE, 0.90

        if meta["image_ratio"] > 0.7:
            return DocumentType.SCANNED_IMAGE, 0.80

        # 2. Scientific: LaTeX-like markers
        scientific_markers = ["∑", "∫", "\\frac", "equation", "theorem", "∀", "∃"]
        if any(m in text for m in scientific_markers):
            return DocumentType.SCIENTIFIC, 0.85

        # 3. Tables: column-like patterns
        table_blocks = [
            b for b in blocks
            if b[4].count("\t") >= 2 or b[4].count("  ") >= 3
        ]
        if len(table_blocks) >= 3:
            return DocumentType.COMPLEX_TABLES, 0.80

        # 4. Default: simple text
        if meta["has_text"] and meta["text_density"] > 0.001:
            return DocumentType.SIMPLE_TEXT, 0.85

        return DocumentType.UNKNOWN, 0.30


class LLMClassifier(BaseClassifier):
    """
    LLM-based classifier (more accurate, slower).
    Sends a text sample to an LLM for classification.
    """

    def __init__(self, config: AppConfig | None = None):
        self._config = config or get_config()
        self._cache = load_cache()

    def classify(self, file_path: Path) -> tuple[DocumentType, float, dict[str, Any]]:
        sample_text = self._extract_sample(file_path)
        key = _hash_text(sample_text)

        if key in self._cache:
            return self._cache[key]

        llm = get_llm(temperature=0.0)
        #structured_llm = llm.with_structured_output(LLMClassificationOutput)

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Document sample:\n\n{sample_text[:2000]}"),
        ])

        result = json.loads(response.content)
        print(result)

        doc_type = DocumentType(result["type"])
        confidence = float(result["confidence"])

        meta = {
            "llm_reason": result.get("reason", ""),
            "sample_used": sample_text[:2000],
        }
        
        self._cache[key] = (doc_type, confidence, meta)
        save_cache(self._cache)

        return doc_type, confidence, meta

    def _extract_sample(self, file_path: Path) -> str:
        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages[:2]:
                text += page.extract_text() or ""
            return text[:2000]
        except Exception:
            return "LLM sample extraction failed"


class ClassifierFactory:
    """Factory for classifiers based on configuration."""

    _registry: dict[str, type[BaseClassifier]] = {
        "heuristic": HeuristicClassifier,
        "llm": LLMClassifier,
    }

    @classmethod
    def create(
        cls,
        strategy: str = "heuristic",
        config: AppConfig | None = None
    ) -> BaseClassifier:
        klass = cls._registry.get(strategy)
        if klass is None:
            raise ValueError(
                f"Unknown strategy: {strategy}. Available: {list(cls._registry)}"
            )

        if strategy == "llm":
            return klass(config=config)

        return klass()

    @classmethod
    def register(cls, name: str, klass: type[BaseClassifier]) -> None:
        """Register a custom classifier without modifying this file."""
        cls._registry[name] = klass


def classifier_node(state: GraphState) -> GraphState:
    """
    LangGraph node: Classifier.
    Determines document type and enriches the state.
    """
    logger.info(f"[Classifier] Processing: {state.file_path.name}")
    
    strategy = os.getenv("CLASSIFIER_STRATEGY", "heuristic")
    classifier = ClassifierFactory.create(strategy)

    try:
        doc_type, confidence, metadata = classifier.classify(state.file_path)

        logger.info(
            f"[Classifier] Detected type: {doc_type} (confidence={confidence:.2f})"
        )

        return state.with_update(
            document_type=doc_type,
            classification_confidence=confidence,
            classification_metadata=metadata,
        )

    except Exception as e:
        logger.error(f"[Classifier] Error: {e}")

        return state.with_update(
            document_type=DocumentType.UNKNOWN,
            classification_confidence=0.0,
            error=str(e),
        )


if __name__ == "__main__":
    state = GraphState(file_path=Path("data/arxiv_doc.pdf"))
    result = classifier_node(state)
