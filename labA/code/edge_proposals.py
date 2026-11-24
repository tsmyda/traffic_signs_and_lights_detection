import math

import cv2
import numpy as np


def generate_edge_proposals(rgb, low_threshold=50, high_threshold=150, min_box_area=16, max_area_fraction=0.25, max_aspect_ratio=10):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image")
    if not 0 <= low_threshold < high_threshold <= 255:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
    if min_box_area < 1 or not math.isfinite(max_area_fraction) or not 0 < max_area_fraction <= 1:
        raise ValueError("Invalid box area limits")
    if not math.isfinite(max_aspect_ratio) or max_aspect_ratio < 1:
        raise ValueError("Maximum aspect ratio must be finite and at least 1")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold, L2gradient=True)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    max_box_area = rgb.shape[0] * rgb.shape[1] * max_area_fraction
    proposals = {}
    for name, mask in (("canny_raw", edges), ("canny_closed", closed)):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if min(width, height) < 2 or not min_box_area <= width * height <= max_box_area:
                continue
            if max(width / height, height / width) > max_aspect_ratio:
                continue
            if cv2.contourArea(contour) < 4:
                continue
            box = (int(x), int(y), int(x + width), int(y + height))
            if box not in proposals:
                proposals[box] = {"box": list(box), "sources": []}
            if name not in proposals[box]["sources"]:
                proposals[box]["sources"].append(name)
    return list(proposals.values())
