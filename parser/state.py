"""
PDFlex Graph State
Defines the state that flows between all nodes in the LangGraph pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from parser.config import DocumentType


class GraphState(BaseModel):
    """
    Immutable state flowing through the LangGraph pipeline.

    Each node receives this state, creates an enriched copy, and returns it.
    This follows immutability principles to simplify debugging and replay.
    """

    # --- Input ---
    file_path: Path = Field(..., description="Absolute path to the PDF file")

    # --- Classifier Output ---
    document_type: Optional[DocumentType] = Field(
        default=None, description="Detected document type"
    )
    classification_confidence: float = Field(
        default=0.0, description="Classification confidence [0-1]"
    )
    classification_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Raw classification metadata"
    )

    # --- Errors ---
    error: Optional[str] = Field(
        default=None, description="Error message if the pipeline fails"
    )

    model_config =  {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }

    def with_update(self, **kwargs) -> "GraphState":
        """
        Returns a new instance with updated fields.
        Ensures immutability between nodes.
        """
        return self.model_copy(update=kwargs)
