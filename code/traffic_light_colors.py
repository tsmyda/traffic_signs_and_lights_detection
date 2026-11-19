import math

import cv2
import numpy as np
from skimage.color import rgb2hsv


COLORS = ("red", "yellow", "green", "unknown")


def validate_config(config):
    for key in ("min_saturation", "min_value", "min_component_fraction"):
        if not math.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f"Invalid color setting: {key}")
    if not isinstance(config["min_component_pixels"], int) or config["min_component_pixels"] < 1:
        raise ValueError("Minimum component size must be a positive integer")
    if not math.isfinite(config["dominance_ratio"]) or config["dominance_ratio"] <= 1:
        raise ValueError("Dominance ratio must be finite and greater than 1")
    if set(config["hue_ranges_degrees"]) != set(COLORS[:3]):
        raise ValueError("Expected hue ranges for red, yellow and green")
    for ranges in config["hue_ranges_degrees"].values():
        if not ranges or any(len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= 360 for bounds in ranges):
            raise ValueError("Invalid hue ranges")


def classify_color(image, box, config):
    left, top = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    right, bottom = min(image.width, math.ceil(box[2])), min(image.height, math.ceil(box[3]))
    scores = {color: 0.0 for color in COLORS[:3]}
    if right <= left or bottom <= top:
        return {"color": "unknown", "color_scores": scores}
    rgb = np.asarray(image.crop((left, top, right, bottom)), dtype=np.float32) / 255
    hsv = rgb2hsv(rgb)
    hue, saturation, value = hsv[:, :, 0] * 360, hsv[:, :, 1], hsv[:, :, 2]
    bright = (saturation >= config["min_saturation"]) & (value >= config["min_value"])
    area = bright.size
    minimum = max(config["min_component_pixels"], math.ceil(config["min_component_fraction"] * area))
    for color, ranges in config["hue_ranges_degrees"].items():
        selected = np.zeros(bright.shape, dtype=bool)
        for lower, upper in ranges:
            selected |= (hue >= lower) & (hue < upper)
        count, labels, stats, _ = cv2.connectedComponentsWithStats((selected & bright).astype(np.uint8), connectivity=8)
        weights = np.bincount(labels.ravel(), weights=(saturation * value).ravel(), minlength=count)
        eligible = stats[1:, cv2.CC_STAT_AREA] >= minimum
        if eligible.any():
            scores[color] = float(weights[1:][eligible].max() / area)
    ordered = sorted(scores, key=scores.get, reverse=True)
    first, second = ordered[:2]
    color = first if scores[first] > 0 and scores[first] >= config["dominance_ratio"] * scores[second] else "unknown"
    return {"color": color, "color_scores": scores}
