import math

import cv2
import numpy as np


def expand_proposals(proposals, image_size, factor):
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("Expansion factor must be finite and at least 1")
    width, height = image_size
    expanded = {}
    for proposal in proposals:
        x1, y1, x2, y2 = proposal["box"]
        margin_x = (x2 - x1) * (factor - 1) / 2
        margin_y = (y2 - y1) * (factor - 1) / 2
        box = (
            max(0, math.floor(x1 - margin_x)),
            max(0, math.floor(y1 - margin_y)),
            min(width, math.ceil(x2 + margin_x)),
            min(height, math.ceil(y2 + margin_y)),
        )
        if box not in expanded:
            expanded[box] = {"box": list(box), "sources": list(proposal["sources"])}
        else:
            expanded[box]["sources"] = sorted(set(expanded[box]["sources"]) | set(proposal["sources"]))
    return list(expanded.values())


def generate_proposals(rgb, delta=5, min_area=16, max_area_fraction=0.25, max_variation=0.25):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB uint8 image")
    height, width = rgb.shape[:2]
    if min(height, width) < 3:
        raise ValueError("Image dimensions must be at least 3 pixels")
    if delta < 1 or min_area < 1 or not 0 < max_area_fraction <= 1 or not 0 < max_variation <= 1:
        raise ValueError("Invalid MSER parameters")
    max_area = max(min_area, round(height * width * max_area_fraction))
    detector = cv2.MSER_create(delta, min_area, max_area, max_variation)
    channels = rgb.astype(np.float32)
    denominator = channels.sum(axis=2)
    normalized = np.zeros((height, width), dtype=np.float32)
    np.divide(np.maximum(channels[:, :, 0], channels[:, :, 2]), denominator, out=normalized, where=denominator > 0)
    inputs = {
        "gray": cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
        "normalized_red_blue": np.rint(normalized * 255).astype(np.uint8),
    }
    proposals = {}
    for name, image in inputs.items():
        _, boxes = detector.detectRegions(image)
        for x, y, box_width, box_height in boxes:
            box = (int(x), int(y), int(x + box_width), int(y + box_height))
            if box not in proposals:
                proposals[box] = {"box": list(box), "sources": []}
            if name not in proposals[box]["sources"]:
                proposals[box]["sources"].append(name)
    return list(proposals.values())
