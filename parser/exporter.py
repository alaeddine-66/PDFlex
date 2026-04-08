"""
PDFlex - Exporter Module
Handles saving the pipeline results to the filesystem.
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path

from parser.state import GraphState
from parser.config import get_config

class BaseExporter(ABC):
    """Abstract contract for output exporters."""
    @abstractmethod
    def export(self, state: GraphState, output_filename: str) -> Path:
        ...

class JSONExporter(BaseExporter):
    """Exports the GraphState to a formatted JSON file."""
    
    def export(self, state: GraphState, output_filename: str) -> Path:
        cfg = get_config()
        output_dir = cfg.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = output_dir / output_filename
        
        state_dict = state.model_dump(exclude={"elements"})
        
        state_dict["file_path"] = str(state_dict["file_path"])
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=4, ensure_ascii=False)
            
        return out_path

class MarkdownExporter(BaseExporter):
    """Exports the extracted text from GraphState to a Markdown file."""

    def export(self, state: GraphState, output_filename: str) -> Path:
        cfg = get_config()
        output_dir = cfg.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_filename.endswith(".md"):
            output_filename = f"{Path(output_filename).stem}.md"

        out_path = output_dir / output_filename

        markdown_content = state.extracted_text or ""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return out_path
