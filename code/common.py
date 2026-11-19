import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np


def valid_box(obj):
    box = obj.get("box2d") or {}
    values = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in values
    ):
        return None
    x1, y1, x2, y2 = values
    return values if x2 > x1 and y2 > y1 else None


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def intersection_over_union(first, second):
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0


def match_predictions(predictions, targets, threshold):
    unmatched = set(range(len(targets)))
    matches = []
    for prediction in sorted(predictions, key=lambda row: row["score"], reverse=True):
        best_index = None
        best_iou = 0
        for index in sorted(unmatched):
            overlap = intersection_over_union(prediction["box"], targets[index])
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        is_true_positive = best_index is not None and best_iou >= threshold
        if is_true_positive:
            unmatched.remove(best_index)
        matches.append({
            **prediction,
            "result": "true_positive" if is_true_positive else "false_positive",
            "matched_target_index": best_index if is_true_positive else None,
            "matched_iou": best_iou if is_true_positive else None,
        })
    true_positive = sum(row["result"] == "true_positive" for row in matches)
    return {
        "true_positive": true_positive,
        "false_positive": len(predictions) - true_positive,
        "false_negative": len(unmatched),
        "matched_predictions": matches,
        "unmatched_targets": [{"target_index": index, "box": targets[index]} for index in sorted(unmatched)],
    }


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_percent": 100 * precision,
        "recall_percent": 100 * recall,
        "f1_percent": 100 * f1,
    }


def ground_truth(path, target_class="traffic_sign"):
    categories = {"traffic_sign": "traffic sign", "traffic_light": "traffic light"}
    if target_class not in categories:
        raise ValueError(f"Unsupported target class: {target_class}")
    annotation = read_json(path)
    frames = annotation.get("frames", [])
    if len(frames) != 1:
        raise ValueError(f"Expected one annotation frame: {path}")
    boxes = []
    invalid = 0
    for obj in frames[0].get("objects", []):
        if obj.get("category") != categories[target_class]:
            continue
        box = valid_box(obj)
        if box is None:
            invalid += 1
        else:
            boxes.append(box)
    return boxes, invalid


def select_images(args):
    if args.image_list.is_file():
        paths = [Path(value) for value in read_json(args.image_list)["images"]]
        if args.images > len(paths):
            raise ValueError("Image list contains fewer images than requested")
        paths = paths[:args.images]
    else:
        candidates = sorted(args.val_dir.glob("*.jpg"))
        if args.images > len(candidates):
            raise ValueError("Not enough validation images")
        paths = random.Random(args.seed).sample(candidates, args.images)
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Duplicate validation image identifiers")
    for path in paths:
        if path.resolve().parent != args.val_dir.resolve():
            raise ValueError(f"Image is outside the validation directory: {path}")
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise ValueError(f"Missing validation image or annotation: {path}")
    return paths


def non_maximum_suppression(candidates, threshold):
    if not candidates:
        return []
    boxes = np.asarray([candidate["box"] for candidate in candidates], dtype=np.float64)
    scores = np.asarray([candidate["score"] for candidate in candidates])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-scores, kind="stable")
    selected = []
    while len(order):
        best = order[0]
        selected.append(candidates[int(best)])
        remaining = order[1:]
        intersection_width = np.maximum(0, np.minimum(boxes[best, 2], boxes[remaining, 2]) - np.maximum(boxes[best, 0], boxes[remaining, 0]))
        intersection_height = np.maximum(0, np.minimum(boxes[best, 3], boxes[remaining, 3]) - np.maximum(boxes[best, 1], boxes[remaining, 1]))
        intersection = intersection_width * intersection_height
        ious = intersection / (areas[best] + areas[remaining] - intersection)
        order = remaining[ious <= threshold]
    return selected
