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
from parser.extractor import get_vision_extractor
from unstructured.partition.pdf import partition_pdf
from markdownify import markdownify as md

def partition_node(state: GraphState):
    logger.info(f"[Partitioner] Partitioning PDF: {state.file_path.name}")

    config = get_config()
    output_dir = config.paths.temp_dir / "temp_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        elements = partition_pdf(
            filename=str(state.file_path),
            strategy="hi_res",
            extract_image_block_types=["Image", "Formula"],
            extract_image_block_output_dir=str(output_dir),
        )
        logger.info(f"[Partitioner] {len(elements)} blocks detected.")
        return state.with_update(elements=elements)

    except Exception as e:
        logger.error(f"[Partitioner] Error: {e}")
        return state.with_update(error=str(e))


def orchestrator_node(state: GraphState) -> GraphState:
    logger.info("[Orchestrator] Processing block by block...")
    final_markdown = []
    vision_extractor = get_vision_extractor()

    for element in state.elements:
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
                    vision_text = vision_extractor.process_single_image(Path(image_path))
                    final_markdown.append(f"\n> **Image Extraction:**\n> {vision_text}\n")
                except Exception as e:
                    logger.error(f"[Orchestrator] AI Vision error on {image_path}: {e}")

    full_text = "\n".join(final_markdown)

    return state.with_update(extracted_text=full_text)

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
