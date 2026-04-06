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
from parser.extractor import UnstructuredExtractor
from parser.exporter import JSONExporter

def partition_node(state: GraphState):
    logger.info(f"[Partitioner] Partitioning PDF: {state.file_path.name}")

    try:

        extractor = UnstructuredExtractor()
        elements = extractor.partition(state.file_path)

        logger.info(f"[Partitioner] {len(elements)} blocks detected.")
        return state.with_update(elements=elements)

    except Exception as e:
        logger.error(f"[Partitioner] Error: {e}")
        return state.with_update(error=str(e))


def orchestrator_node(state: GraphState) -> GraphState:
    logger.info("[Orchestrator] Processing block by block...")
    try:

        extractor = UnstructuredExtractor()
        full_text, metadata = extractor.extract(state.elements)

        return state.with_update(extracted_text=full_text, extracted_metadata=metadata)

    except Exception as e:
        logger.error(f"[Orchestrator] Error: {e}")
        return state.with_update(error=str(e))


def build_graph() -> StateGraph:
    """
    Builds and compiles the PDFlex LangGraph.
    """
    graph = StateGraph(GraphState)

    graph.add_node("partitioner", partition_node)
    graph.add_node("orchestrator", orchestrator_node)

    graph.set_entry_point("partitioner")
    graph.add_edge("partitioner", "orchestrator")

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
