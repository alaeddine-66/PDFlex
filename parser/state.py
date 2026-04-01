"""
PDFlex Graph State
Defines the state that flows between all nodes in the LangGraph pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

class GraphState(BaseModel):
    """
    Immutable state flowing through the LangGraph pipeline.

    Each node receives this state, creates an enriched copy, and returns it.
    """

    file_path: Path = Field(..., description="Absolute path to the PDF file")

    extracted_text: Optional[str] = Field(
        default=None, description="Text extracted by the worker"
    )

    extracted_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata (e.g., number of pages, tables, etc.)"
    )

    elements: list[Any] = Field(
            default_factory=list, description="List of parsed elements from the document"
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
