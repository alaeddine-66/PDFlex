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

    def sort_elements(self, elements: list[Any]) -> list[Any]:
        """
        Sorts document elements by columns and reading order.
        
        Args:
            elements: List of Unstructured elements.
            
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

            page_min_x = min(self._get_x_bounds(e)[0] for e in els_with_coords)
            page_max_x = max(self._get_x_bounds(e)[1] for e in els_with_coords)
            page_width = page_max_x - page_min_x

            clean_els_with_coords = []
            for e in els_with_coords:
                x0, x1 = self._get_x_bounds(e)
                if x1 < page_min_x + (page_width * 0.08):
                    continue
                clean_els_with_coords.append(e)

            els_with_coords = clean_els_with_coords
            
            if not els_with_coords:
                continue
            
            page_min_x = min(self._get_x_bounds(e)[0] for e in els_with_coords)
            page_max_x = max(self._get_x_bounds(e)[1] for e in els_with_coords)
            page_width = page_max_x - page_min_x

            # Step 1: Isolate wide elements (like main titles)
            spanning_els, normal_els = self._extract_spanners(els_with_coords, page_min_x, page_width, threshold=0.7)

            # Step 2: Find the anchors (column beginnings)
            anchors = self._detect_columns_by_anchors(normal_els, page_width)

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
    def _extract_spanners(elements: list[Any],page_min_x: float, page_width: float, threshold: float = 0.7) -> tuple[list[Any], list[Any]]:
        """Isolates wide elements OR perfectly centered elements"""
        if not elements:
            return [], []

        page_center = page_min_x + (page_width / 2)
            
        spanning_elements, normal_elements = [], []
        for e in elements:
            min_x, max_x = UnstructuredLayoutAnalyzer._get_x_bounds(e)
            width = max_x - min_x
            
            is_wide = width > (threshold * page_width)
            
            element_center = (min_x + max_x) / 2
            is_centered = abs(element_center - page_center) < (page_width * 0.05)

            if is_wide or is_centered: 
                spanning_elements.append(e)
            else:
                normal_elements.append(e)
                
        return spanning_elements, normal_elements

    @staticmethod
    def _calculate_dynamic_tolerance(x_mins: list[float], page_width: float) -> float:
        """
        Calculates a dynamic indentation tolerance based on column gaps.
        Analyzes the spaces between left-aligned elements to differentiate
        between text indentation and actual column separators.
        """
        if len(x_mins) < 2:
            return page_width * 0.05

        gaps = [x_mins[i] - x_mins[i-1] for i in range(1, len(x_mins))]
        max_gap = max(gaps)
        
        if max_gap > (page_width * 0.08):
            return max_gap * 0.6  
            
        return page_width * 0.05

    @staticmethod
    def _detect_columns_by_anchors(elements: list[Any], page_width: float) -> list[float]:
        """Finds 'Anchors' based on dominant left-aligned points (X-min clustering)."""
        if not elements:
            return []

        reliable_types = ["NarrativeText", "ListItem", "Text"]
        anchor_elements = []
        for e in elements:
            if type(e).__name__ in reliable_types:
                text_lower = str(e.text).strip().lower()
                if not text_lower.startswith(("figure", "table", "fig.", "tab.")):
                    anchor_elements.append(e)

        if not anchor_elements:
            anchor_elements = elements 

        x_mins = sorted([UnstructuredLayoutAnalyzer._get_x_bounds(e)[0] for e in anchor_elements])
        tolerance = UnstructuredLayoutAnalyzer._calculate_dynamic_tolerance(x_mins, page_width)
        print(f"{tolerance=}")

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
        """
        Reassembles the page by placing Top Spanners first, 
        then reading Columns strictly from Left to Right, 
        and finally placing Bottom Spanners.
        """
        valid_columns = [col for col in columns_els if col]
        for col in valid_columns:
            col.sort(key=UnstructuredLayoutAnalyzer._get_y)
            
        spanning_els.sort(key=UnstructuredLayoutAnalyzer._get_y)

        valid_columns.sort(key=lambda col: UnstructuredLayoutAnalyzer._get_x_bounds(col[0])[0])

        if valid_columns:
            columns_start_y = min(UnstructuredLayoutAnalyzer._get_y(col[0]) for col in valid_columns)
        else:
            columns_start_y = float('inf')

        top_spanners = [e for e in spanning_els if UnstructuredLayoutAnalyzer._get_y(e) <= columns_start_y + 10]
        bottom_spanners = [e for e in spanning_els if UnstructuredLayoutAnalyzer._get_y(e) > columns_start_y + 10]

        final_elements = []
        
        final_elements.extend(top_spanners)
        
        for col in valid_columns:
            final_elements.extend(col)
            
        final_elements.extend(bottom_spanners)

        return final_elements
