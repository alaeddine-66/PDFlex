"""
PDFlex - Main LangGraph Graph
Orchestrates the complete pipeline.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger
from langgraph.graph import StateGraph

from parser.config import AppConfig, get_config
from parser.state import GraphState
from parser.unstructured_extractor import UnstructuredExtractor
from parser.exporter import JSONExporter, MarkdownExporter

def extract_node(state: GraphState) -> GraphState:
    logger.info("[Orchestrator] Processing block by block...")
    try:

        extractor = UnstructuredExtractor()
        full_text, metadata = extractor.extract(state.file_path)

        return state.with_update(extracted_text=full_text, extracted_metadata=metadata)

    except Exception as e:
        logger.error(f"[Orchestrator] Error: {e}")
        return state.with_update(error=str(e))


def build_graph() -> StateGraph:
    """
    Builds and compiles the PDFlex LangGraph.
    """
    graph = StateGraph(GraphState)

    graph.add_node("extract", extract_node)

    graph.set_entry_point("extract")

    return graph.compile()


class PDFlexPipeline:
    """
    Main entry point for PDFlex.
    Encapsulates the LangGraph and exposes a simple API.
    """

    def __init__(self):
        self._graph = build_graph()

    def run(self, file_path: str | Path) -> GraphState:
        """
        Executes the complete pipeline on a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            The final GraphState with extracted text and metadata.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported format: {path.suffix}. Only .pdf files are accepted.")

        logger.info(f"[PDFlex] Starting pipeline for: {path.name}")

        initial_state = GraphState(file_path=path)
        final_state = self._graph.invoke(initial_state)
        final_state = GraphState(**final_state)

        exporter = JSONExporter()
        output_filename = f"{path.stem}.json"
        exporter.export(final_state, output_filename)

        md_exporter = MarkdownExporter()
        md_exporter.export(final_state, f"{path.stem}.md")

        logger.info("[PDFlex] Pipeline finished")
        return final_state

    def stream(self, file_path: str | Path):
        """
        Streaming version: yields intermediate states of each node.
        Useful for real-time monitoring.
        """
        path = Path(file_path).resolve()
        initial_state = GraphState(file_path=path)

        for event in self._graph.stream(initial_state):
            node_name = list(event.keys())[0]
            logger.debug(f"[Stream] Node '{node_name}' finished")
            yield event
