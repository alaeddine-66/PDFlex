"""
PDFlex - Layout Analysis and Sorting
Handles complex document structures using Unstructured.io elements.
"""
from collections import defaultdict
from typing import Any


class UnstructuredLayoutAnalyzer:
    """
    Analyzes and sorts Unstructured document elements to reconstruct
    the natural reading order, handling multi-column layouts.
    """

    def sort_elements(self, elements: list[Any], indent_tolerance: float = 30.0) -> list[Any]:
        """
        Sorts document elements by columns and reading order.
        
        Args:
            elements: List of Unstructured elements.
            indent_tolerance: Pixel tolerance for grouping indented text into the same column.
            
        Returns:
            A new list of sorted elements.
        """
        sorted_elements = []
        pages = self._group_by_page(elements)

        for page_num in sorted(pages.keys()):
            page_els = pages[page_num]

            els_with_coords = [e for e in page_els if e.metadata.coordinates and e.metadata.coordinates.points]
            els_without_coords = [e for e in page_els if not (e.metadata.coordinates and e.metadata.coordinates.points)]

            if not els_with_coords:
                sorted_elements.extend(page_els)
                continue

            # Step 1: Isolate wide elements (like main titles)
            spanning_els, normal_els = self._extract_spanners(els_with_coords, threshold=0.7)

            # Step 2: Find the anchors (column beginnings)
            anchors = self._detect_columns_by_anchors(normal_els, indent_tolerance)

            # Step 3: Assign paragraphs to their respective columns
            cols_elements = self._assign_to_anchors(normal_els, anchors)

            # Step 4: Reassemble the page top-to-bottom
            page_sorted = self._reassemble_page(spanning_els, cols_elements)

            # Append metadata/coordinate-less items first, then the sorted page
            sorted_elements.extend(els_without_coords)
            sorted_elements.extend(page_sorted)

        return sorted_elements

    @staticmethod
    def _get_y(e: Any) -> float:
        """Returns the highest Y coordinate of an element."""
        if not e.metadata.coordinates or not e.metadata.coordinates.points:
            return 0.0
        return min(pt[1] for pt in e.metadata.coordinates.points)

    @staticmethod
    def _get_x_bounds(e: Any) -> tuple[float, float]:
        """Returns the (x_min, x_max) bounding coordinates of an element."""
        pts = e.metadata.coordinates.points
        return min(pt[0] for pt in pts), max(pt[0] for pt in pts)

    @staticmethod
    def _group_by_page(elements: list[Any]) -> dict[int, list[Any]]:
        """Groups elements by their page number."""
        pages = defaultdict(list)
        for el in elements:
            page_num = el.metadata.page_number or 1
            pages[page_num].append(el)
        return pages

    @staticmethod
    def _extract_spanners(elements: list[Any], threshold: float = 0.7) -> tuple[list[Any], list[Any]]:
        """Isolates wide elements (e.g., > 70% of the page width) like main titles."""
        if not elements:
            return [], []
            
        page_min_x = min(UnstructuredLayoutAnalyzer._get_x_bounds(e)[0] for e in elements)
        page_max_x = max(UnstructuredLayoutAnalyzer._get_x_bounds(e)[1] for e in elements)
        page_width = page_max_x - page_min_x

        spanning_elements, normal_elements = [], []
        for e in elements:
            min_x, max_x = UnstructuredLayoutAnalyzer._get_x_bounds(e)
            if (max_x - min_x) > threshold * page_width:
                spanning_elements.append(e)
            else:
                normal_elements.append(e)
                
        return spanning_elements, normal_elements

    @staticmethod
    def _detect_columns_by_anchors(elements: list[Any], tolerance: float = 30.0) -> list[float]:
        """Finds 'Anchors' based on dominant left-aligned points (X-min clustering)."""
        if not elements:
            return []

        x_mins = sorted([UnstructuredLayoutAnalyzer._get_x_bounds(e)[0] for e in elements])

        anchors = []
        current_cluster = [x_mins[0]]

        for x in x_mins[1:]:
            # If the left edge is close to the start of our cluster (accounts for indentation)
            if x - current_cluster[0] <= tolerance:
                current_cluster.append(x)
            else:
                # Validate the anchor (average of the cluster) and start a new one
                anchors.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [x]

        if current_cluster:
            anchors.append(sum(current_cluster) / len(current_cluster))

        return anchors

    @staticmethod
    def _assign_to_anchors(elements: list[Any], anchors: list[float]) -> list[list[Any]]:
        """Assigns each element to the closest anchor (column)."""
        columns = [[] for _ in range(len(anchors))]

        for e in elements:
            x_min, _ = UnstructuredLayoutAnalyzer._get_x_bounds(e)

            closest_idx = 0
            min_dist = float('inf')

            for i, anchor in enumerate(anchors):
                dist = abs(x_min - anchor)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            columns[closest_idx].append(e)

        return columns

    @staticmethod
    def _reassemble_page(spanning_els: list[Any], columns_els: list[list[Any]]) -> list[Any]:
        """Sorts columns vertically and reassembles the full page sequentially."""
        # 1. Sort contents inside each column from top to bottom
        for col in columns_els:
            col.sort(key=UnstructuredLayoutAnalyzer._get_y)
        spanning_els.sort(key=UnstructuredLayoutAnalyzer._get_y)

        blocks = []
        
        # 2. Treat spanning elements as independent blocks
        for e in spanning_els:
            blocks.append(("spanner", UnstructuredLayoutAnalyzer._get_y(e), [e]))

        # 3. Treat columns as independent blocks
        for col in columns_els:
            if col:
                blocks.append(("column", UnstructuredLayoutAnalyzer._get_y(col[0]), col))

        # 4. Sort blocks by their starting Y coordinate
        blocks.sort(key=lambda b: b[1])

        final_elements = []
        for _, _, els in blocks:
            final_elements.extend(els)

        return final_elements
