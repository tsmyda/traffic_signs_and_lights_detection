"""Wariant H1: przestrojenie parametrów klasycznej metody koloru (Lab A) na boksach YOLO z val.

Bierze TP-światła z val (YOLO przy zamrożonym progu), przeszukuje siatkę parametrów HSV
i zapisuje najlepszy config. Test nie jest używany.

    python labB/code/tune_hsv.py --predictions labB/runs/predictions/yolo_coco_1280_val1000.jsonl \
        --threshold-from labB/results/yolo_coco_1280_val_light.json
"""
import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from paths import CONFIGS, IMAGES, LAB_A
from bdd import COLORS, load_annotation, read_jsonl
from common import match_predictions  # Lab A
from traffic_light_colors import classify_color  # Lab A
from hybrid_color import shrink

GRID = {
    "min_saturation": [0.3, 0.4, 0.5],
    "min_value": [0.4, 0.5, 0.6],
    "dominance_ratio": [1.1, 1.2, 1.5],
    "red_yellow": [15, 20, 25],      # granica hue red/yellow [deg]
    "yellow_green": [65, 75, 85],    # granica hue yellow/green [deg]
    "roi_shrink": [0.0, 0.1],        # obcięcie ROI z każdej strony (ułamek boku)
}


def collect(rows, threshold):
    samples = []
    for row in rows:
        annotation = load_annotation("val", row["image_id"])
        lights = [p for p in row["predictions"] if p["class"].startswith("traffic_light") and p["score"] >= threshold]
        result = match_predictions(lights, annotation["traffic_light"], 0.5)
        tps = [m for m in result["matched_predictions"] if m["result"] == "true_positive"]
        if not tps:
            continue
        with Image.open(IMAGES / "val" / row["image_filename"]) as source:
            image = source.convert("RGB")
        for match in tps:
            x1, y1, x2, y2 = [int(v) for v in match["box"]]
            crop = image.crop((x1, y1, x2 + 1, y2 + 1))
            local = [match["box"][0] - x1, match["box"][1] - y1, match["box"][2] - x1, match["box"][3] - y1]
            samples.append((crop, local, annotation["light_colors"][match["matched_target_index"]]))
    return samples


def make_config(base, values):
    config = copy.deepcopy(base)
    config.update({k: values[k] for k in ("min_saturation", "min_value", "dominance_ratio")})
    config["hue_ranges_degrees"] = {"red": [[0, values["red_yellow"]], [340, 360]],
                                    "yellow": [[values["red_yellow"], values["yellow_green"]]],
                                    "green": [[values["yellow_green"], 175]]}
    config["roi_shrink"] = values["roi_shrink"]
    return config


def score(samples, config, objective):
    confusion = np.zeros((4, 4), dtype=int)
    for crop, box, truth in samples:
        predicted = classify_color(crop, shrink(box, config["roi_shrink"]), config)["color"]
        confusion[COLORS.index(truth), COLORS.index(predicted)] += 1
    if objective == "accuracy":
        return confusion.trace() / confusion.sum()
    f1 = [2 * confusion[i, i] / (confusion[i].sum() + confusion[:, i].sum() or 1) for i in range(4)]
    return float(np.mean(f1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="predykcje YOLO na val1000")
    parser.add_argument("--threshold-from", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=LAB_A / "configs/traffic_light_colors.json")
    parser.add_argument("--objective", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--output", type=Path, default=CONFIGS / "traffic_light_colors_yolo.json")
    args = parser.parse_args()

    threshold = json.loads(args.threshold_from.read_text(encoding="utf-8"))["score_threshold"]
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    samples = collect(read_jsonl(args.predictions), threshold)
    print(f"{len(samples)} TP lights on val")
    baseline = score(samples, {**base, "roi_shrink": 0.0}, args.objective)
    best, best_score = None, -1
    for combo in itertools.product(*GRID.values()):
        config = make_config(base, dict(zip(GRID, combo)))
        value = score(samples, config, args.objective)
        if value > best_score:
            best, best_score = config, value
            print(f"{args.objective}={value:.4f} {dict(zip(GRID, combo))}", flush=True)
    args.output.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    print(f"Lab A config: {baseline:.4f}; tuned: {best_score:.4f}; saved {args.output}")


if __name__ == "__main__":
    main()
