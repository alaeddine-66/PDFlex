"""
PDFlex - Unstructured Extraction Worker
"""
import re
from pathlib import Path
from typing import Any

import pdfplumber
from PIL import Image
from markdownify import markdownify as md
from unstructured.partition.pdf import partition_pdf
from loguru import logger
from collections import defaultdict

from parser.base import BaseExtractor
from parser.vision import get_vision_extractor
from parser.layout_analyzer import UnstructuredLayoutAnalyzer
from parser.config import get_config

def clean_text(text: str) -> str:
    """Cleans text artifacts like end-of-line hyphens."""
    if not text:
        return ""
    cleaned = re.sub(r'([a-zA-Z]+)-\s+([a-zA-Z]+)', r'\1\2', str(text))
    return cleaned.strip()

def is_garbled(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 5:
        return False

    single_chars = [w for w in words if len(w) == 1 and w.isalpha() and w.lower() not in ['a', 'i']]

    ratio = len(single_chars) / len(words)
    return ratio > 0.25

def is_garbled(text: str) -> bool:
    """
    Checks if the extracted text contains an abnormally high ratio of single characters.
    """
    if not text:
        return False
    
    text_no_spaces = text.replace(" ", "")
    if len(text_no_spaces) > 0:
        symbol_count = sum(1 for char in text_no_spaces if char in "()[]{}=+-*/><$")
        if (symbol_count / len(text_no_spaces)) > 0.15:
            return False 

    words = text.split()
    if len(words) < 5:
        return False

    single_chars = [w for w in words if len(w) == 1 and w.isalpha() and w.lower() not in ['a', 'i']]
    ratio = len(single_chars) / len(words)
    
    return ratio > 0.25

class UnstructuredExtractor(BaseExtractor):
    """
    PDF extractor utilizing Unstructured.io.
    """

    def __init__(self):
        """
        Initializes the layout analyzer, vision extractor, and application config.
        """
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
            strategy="hi_res",
            extract_image_block_types=["Image", "Formula", "Table"],
            extract_image_block_to_payload=True,
        )

        logger.info(f"[UnstructuredExtractor] {len(elements)} blocks detected. Applying layout sorting...")
        sorted_elements = self.layout_analyzer.sort_elements(elements)

        return sorted_elements

    def extract(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """
        Partitions the PDF and converts the parsed Unstructured elements into a formatted Markdown string.
        This overrides the BaseExtractor interface.

        Args:
            file_path (Path): Path to the original PDF file.

        Returns:
            tuple[str, dict[str, Any]]: The final Markdown string and metadata dictionary.
        """
        elements = self.partition(file_path)
        elements = self._filter_overlapping_elements(elements)

        final_markdown = []
        current_index = 0
        
        while current_index < len(elements):
            element = elements[current_index]
            element_type = type(element).__name__

            if element_type in ["Text", "NarrativeText", "Title", "ListItem", "FigureCaption"]:
                extracted_texts, new_index = self._process_text_element(elements, current_index, file_path)
                final_markdown.extend(extracted_texts)
                current_index = new_index

            elif element_type == "Table":
                table_text = self._process_table_element(element)
                if table_text:
                    final_markdown.append(table_text)
                current_index += 1

            elif element_type in ["Image", "Figure", "Formula"]:
                media_text = self._process_media_element(element)
                if media_text:
                    final_markdown.append(media_text)
                current_index += 1

            elif element_type in ["Header", "Footer", "PageBreak", "PageIndicator"]:
                current_index += 1

            else:
                logger.error(f"[UnstructuredExtractor] Element of type {element_type} is not supported.")
                current_index += 1

        full_text = "\n".join(final_markdown)
        metadata = {
            "engine": "unstructured",
            "blocks_detected": len(elements)
        }

        return full_text, metadata

    def _process_text_element(self, elements: list[Any], start_index: int, file_path: Path) -> tuple[list[str], int]:
        """
        Handles standard text extraction and routes to visual healing if the text is garbled.

        Returns:
            tuple[list[str], int]: A list containing the markdown strings, and the next index to process.
        """
        element = elements[start_index]
        element_type = type(element).__name__
        cleaned_text = clean_text(element.text)
        coords = getattr(element.metadata, "coordinates", None)
        
        if is_garbled(cleaned_text) and coords and coords.points:
            return self._heal_garbled_text_blocks(elements, start_index, file_path)

        prefix = "## " if element_type == "Title" else ""
        extracted_content = [f"{prefix}{cleaned_text}\n"]
        
        return extracted_content, start_index + 1

    def _heal_garbled_text_blocks(self, elements: list[Any], start_index: int, file_path: Path) -> tuple[list[str], int]:
        """
        Groups adjacent garbled text blocks, crops the corresponding area from the PDF, 
        and extracts the text using the Vision LLM.

        Returns:
            tuple[list[str], int]: The healed text content, and the next index to process in the main loop.
        """
        initial_element = elements[start_index]
        page_num = getattr(initial_element.metadata, "page_number", 1)
        
        logger.warning(f"[UnstructuredExtractor] Garbled text detected on page {page_num}. Initiating visual healing...")

        current_index = start_index
        group_coords = []
        extracted_content = []

        #Group contiguous garbled blocks on the same page
        while current_index < len(elements):
            next_el = elements[current_index]
            next_type = type(next_el).__name__
            next_page = getattr(next_el.metadata, "page_number", 1)
            next_coords = getattr(next_el.metadata, "coordinates", None)

            is_valid_type = next_type in ["Text", "NarrativeText", "Title", "ListItem", "FigureCaption"]
            
            if is_valid_type and next_page == page_num and next_coords and next_coords.points:
                if is_garbled(clean_text(next_el.text)):
                    group_coords.append(next_coords)
                    current_index += 1
                    continue
            break 

        blocks_grouped = current_index - start_index
        logger.warning(f"[UnstructuredExtractor] Grouping {blocks_grouped} garbled blocks on page {page_num}...")
        
        #Crop the PDF and process with Vision LLM
        try:
            with pdfplumber.open(file_path) as pdf:
                page = pdf.pages[page_num - 1]

                pdf_w, pdf_h = float(page.width), float(page.height)
                
                uns_w = group_coords[0].system.width
                uns_h = group_coords[0].system.height
                
                scale_x = pdf_w / uns_w
                scale_y = pdf_h / uns_h

                x_coords = [p[0] * scale_x for c in group_coords for p in c.points]
                y_coords = [p[1] * scale_y for c in group_coords for p in c.points]

                padding = 5
                x0 = max(0, min(x_coords) - padding)
                y0 = max(0, min(y_coords) - padding)
                x1 = min(pdf_w, max(x_coords) + padding)
                y1 = min(pdf_h, max(y_coords) + padding)

                bounding_box = (x0, y0, x1, y1)
                cropped_page = page.crop(bounding_box)

                crop_path = self.config.paths.temp_dir / f"garbled_crop_p{page_num}_{int(y0)}.jpg"
                cropped_image = cropped_page.to_image(resolution=300)
                cropped_image.save(str(crop_path))

                healed_text = self.vision_extractor.process_single_image(crop_path)
                extracted_content.append(healed_text)
                logger.debug(f"Original garbled text: '{initial_element.text}' -> Healed: '{healed_text}'")

            logger.info(f"[UnstructuredExtractor] Visual healing completed successfully!")
            return extracted_content, current_index

        except Exception as e:
            logger.error(f"[UnstructuredExtractor] Visual healing failed: {e}")
            # Fallback: return the original garbled text if healing fails
            for k in range(start_index, current_index):
                fallback_text = clean_text(elements[k].text)
                extracted_content.append(f"{fallback_text}\n")
            
            return extracted_content, current_index

    def _process_table_element(self, element: Any) -> str:
        """
        Processes table elements, prioritizing HTML format, with a fallback to Vision LLM.
        """
        html_table = getattr(element.metadata, "text_as_html", None)
        base64_data = getattr(element.metadata, "image_base64", None)

        if html_table:
            markdown_table = md(html_table)
            return f"{markdown_table}\n"

        elif base64_data:
            logger.info(f"[UnstructuredExtractor] HTML not found. Triggering Vision LLM fallback for table")
            try:
                vision_text = self.vision_extractor.process_single_image_from_bytes(base64_data)
                return f"\n{vision_text}\n"
            except Exception as e:
                logger.error(f"[UnstructuredExtractor] Vision LLM table extraction error: {e}")
                return f"{element.text}\n"

        return f"{element.text}\n"

    def _process_media_element(self, element: Any) -> str:
        """
        Processes graphical elements (Images, Figures, Formulas) using the Vision LLM.
        Filters out visual artifacts based on dimensions.
        """
        base64_data = getattr(element.metadata, "image_base64", None)
        
        if base64_data:
            try:
                x0, y0, x1, y1 = self._get_bounding_box(element)
                width = x1 - x0
                height = y1 - y0

                # Filter out artifacts: extremely small images or thin layout lines
                if (width < 10) or (height < 10) or (width / height > 10) or (height / width > 10):
                    logger.debug(f"[UnstructuredExtractor] Ignored visual artifact (dimensions: {width}x{height})")
                    return ""

                # Process valid images through the Vision extractor
                vision_text = self.vision_extractor.process_single_image_from_bytes(base64_data)
                return f"\n> **Image Extraction:**\n> {vision_text}\n"

            except Exception as e:
                logger.error(f"[UnstructuredExtractor] Vision LLM media extraction error : {e}")
        
        return ""

    def _get_bounding_box(self, element: Any) -> tuple[float, float, float, float] | None:
        """
        Extracts the (x0, y0, x1, y1) bounding box from an Unstructured element.
        """
        coords = getattr(element.metadata, "coordinates", None)
        if not coords or not coords.points:
            return None
        
        points = coords.points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        return min(xs), min(ys), max(xs), max(ys)

    def _calculate_overlap_ratio(self, box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
        """
        Calculates the overlap ratio between two bounding boxes.
        Returns a float between 0.0 and 1.0.
        """
        x0_1, y0_1, x1_1, y1_1 = box1
        x0_2, y0_2, x1_2, y1_2 = box2

        # Find intersection coordinates
        intersect_x0 = max(x0_1, x0_2)
        intersect_y0 = max(y0_1, y0_2)
        intersect_x1 = min(x1_1, x1_2)
        intersect_y1 = min(y1_1, y1_2)

        # Check if there is an intersection
        if intersect_x1 <= intersect_x0 or intersect_y1 <= intersect_y0:
            return 0.0

        intersect_area = (intersect_x1 - intersect_x0) * (intersect_y1 - intersect_y0)
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

        min_area = min(area1, area2)
        if min_area == 0:
            return 0.0

        return intersect_area / min_area

    def _filter_overlapping_elements(self, elements: list[Any]) -> list[Any]:
        """
        Filters out text elements that heavily overlap with media/table elements.
        """
        media_types = ["Image", "Figure", "Formula", "Table"]
        media_by_page = defaultdict(list)
        for el in elements:
            if type(el).__name__ in media_types:
                page_num = getattr(el.metadata, "page_number", 1)
                media_by_page[page_num].append(el)
        
        filtered_elements = []

        for el in elements:
            el_type = type(el).__name__
            
            # Keep all media elements
            if el_type in media_types:
                filtered_elements.append(el)
                continue

            el_box = self._get_bounding_box(el)
            if not el_box:
                filtered_elements.append(el)
                continue

            is_duplicate = False
            el_page = getattr(el.metadata, "page_number", 1)
            page_medias = media_by_page.get(page_num, [])

            for media_el in page_medias:
                media_box = self._get_bounding_box(media_el)
                if not media_box:
                    continue

                overlap = self._calculate_overlap_ratio(el_box, media_box)
                
                if overlap > 0.6:
                    logger.debug(f"[UnstructuredExtractor] Removed overlapping text block on page {el_page} (Overlap: {overlap:.0%})")
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_elements.append(el)

        return filtered_elements
