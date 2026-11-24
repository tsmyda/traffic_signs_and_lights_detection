"""Ewaluacja detekcji i koloru - identyczna logika jak w Lab A (match_predictions z labA/code/common.py).

- P/R/F1 przy IoU >= 0.5, dopasowanie zachłanne po score, 1:1, duplikaty = FP.
- Kolor tylko dla poprawnie zlokalizowanych świateł (TP), macierz 4x4 (wiersze = GT, kolumny = predykcja).
- Dopasowanie liczone raz dla wszystkich detekcji; dla progu t wynik = detekcje o score >= t
  (zachłanne dopasowanie po malejącym score -> decyzja dla detekcji zależy tylko od wyższych score'ów,
  więc sweep progu jest dokładny, tak jak w Lab A).

Przykłady:
    # strojenie progu na val (best F1)
    python labB/code/evaluate.py --predictions labB/runs/predictions/yolo_coco_1280_val1000.jsonl \
        --split val --target traffic_light --sweep --name yolo_coco_1280_val_light
    # test z progiem zamrożonym na val
    python labB/code/evaluate.py --predictions labB/runs/predictions/hybrid_H0_test.jsonl \
        --split test --target traffic_light --threshold-from labB/results/yolo_coco_1280_val_light.json \
        --name hybrid_H0_test
    # kontrola: Lab A musi odtworzyć assets/test_results.json
    python labB/code/evaluate.py --predictions labA/predictions/traffic_light_colors.jsonl \
        --split test --target traffic_light --threshold 0 --name labA_light_test
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from paths import RESULTS, RUNS
from bdd import COLORS, load_annotation, read_jsonl
from common import match_predictions, metrics  # Lab A

IOU = 0.5
SIZE_BINS = [(0, 8), (8, 16), (16, 32), (32, 10_000)]  # krótszy bok boksu w px (jak w raporcie A)

# która klasa predykcji jest oceniana względem której klasy GT
TARGETS = {
    "traffic_light": ("traffic_light", lambda c: c.startswith("traffic_light")),
    "traffic_sign": ("traffic_sign", lambda c: c == "traffic_sign"),
    "stop_sign": ("traffic_sign", lambda c: c == "stop_sign"),  # COCO stop sign vs wszystkie znaki BDD
}


def size_bin(box):
    side = min(box[2] - box[0], box[3] - box[1])
    return next(f"{lo}-{hi}" if hi < 10_000 else f">={lo}" for lo, hi in SIZE_BINS if lo <= side < hi)


def predicted_color(prediction):
    if "color" in prediction:
        return prediction["color"]
    suffix = prediction["class"].removeprefix("traffic_light_")
    return suffix if suffix in COLORS else None


def match_all(rows, split, target):
    """Jedno dopasowanie na obraz dla wszystkich detekcji. Zwraca płaskie listy detekcji i obiektów GT."""
    gt_key, accept = TARGETS[target]
    detections, objects, seconds = [], [], []
    for row in rows:
        annotation = load_annotation(split, row["image_id"])
        targets = annotation[gt_key]
        predictions = [p for p in row["predictions"] if accept(p["class"])]
        result = match_predictions(predictions, targets, IOU)
        base = len(objects)
        for index, box in enumerate(targets):
            objects.append({"image_id": row["image_id"], "index": index, "size": size_bin(box),
                            "timeofday": annotation["timeofday"],
                            "color": annotation["light_colors"][index] if gt_key == "traffic_light" else None,
                            "matched_score": None, "predicted_color": None})
        for match in result["matched_predictions"]:
            detection = {"score": match["score"], "tp": match["result"] == "true_positive",
                         "size": size_bin(match["box"]), "timeofday": annotation["timeofday"]}
            if detection["tp"]:
                gt = objects[base + match["matched_target_index"]]
                gt["matched_score"], gt["predicted_color"] = match["score"], predicted_color(match)
                detection["size"] = gt["size"]
            detections.append(detection)
        if "seconds" in row:
            seconds.append(row["seconds"] + row.get("color_seconds", 0.0))
    return detections, objects, seconds


def at_threshold(detections, objects, threshold):
    tp = sum(d["tp"] and d["score"] >= threshold for d in detections)
    fp = sum((not d["tp"]) and d["score"] >= threshold for d in detections)
    return metrics(tp, fp, len(objects) - tp)


def sweep(detections, objects):
    scores = np.array([d["score"] for d in detections])
    is_tp = np.array([d["tp"] for d in detections], dtype=bool)
    order = np.argsort(-scores, kind="stable")
    scores, is_tp = scores[order], is_tp[order]
    tp, fp = np.cumsum(is_tp), np.cumsum(~is_tp)
    last = np.r_[scores[1:] != scores[:-1], True]  # pełne grupy remisów
    tp, fp, thresholds = tp[last], fp[last], scores[last]
    f1 = 2 * tp / (2 * tp + fp + (len(objects) - tp))
    best = int(np.argmax(f1))
    curve = [{"score_threshold": float(t), "precision": float(a / (a + b)), "recall": float(a / len(objects))}
             for t, a, b in zip(thresholds[::max(1, len(thresholds) // 200)], tp[::max(1, len(tp) // 200)],
                                fp[::max(1, len(fp) // 200)])]
    return float(thresholds[best]), curve


def breakdown(detections, objects, threshold, key):
    groups = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    for d in detections:
        if d["score"] >= threshold:
            groups[d[key]][0 if d["tp"] else 1] += 1
    for o in objects:
        if o["matched_score"] is None or o["matched_score"] < threshold:
            groups[o[key]][2] += 1
    return {name: metrics(*values) for name, values in sorted(groups.items())}


def color_summary(objects, threshold):
    confusion = np.zeros((4, 4), dtype=int)
    for o in objects:
        if o["matched_score"] is not None and o["matched_score"] >= threshold and o["predicted_color"]:
            confusion[COLORS.index(o["color"]), COLORS.index(o["predicted_color"])] += 1
    evaluated = int(confusion.sum())
    if not evaluated:
        return None
    per_class = {}
    for i, color in enumerate(COLORS):
        support, predicted, correct = confusion[i].sum(), confusion[:, i].sum(), confusion[i, i]
        per_class[color] = {"support": int(support),
                            "precision": float(correct / predicted) if predicted else 0.0,
                            "recall": float(correct / support) if support else 0.0}
    return {"correctly_localized_lights": evaluated, "correct_colors": int(confusion.trace()),
            "color_accuracy_percent": 100 * confusion.trace() / evaluated, "per_class": per_class,
            "confusion_labels": list(COLORS), "confusion_matrix": confusion.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--target", choices=list(TARGETS), required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sweep", action="store_true", help="wybierz próg best-F1 (tylko val!)")
    group.add_argument("--threshold", type=float)
    group.add_argument("--threshold-from", type=Path, help="JSON z wynikiem --sweep na val")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.sweep and args.split == "test":
        parser.error("Progu nie wolno dobierać na teście")

    rows = read_jsonl(args.predictions)
    detections, objects, seconds = match_all(rows, args.split, args.target)
    curve = None
    if args.sweep:
        threshold, curve = sweep(detections, objects)
    elif args.threshold_from:
        threshold = json.loads(args.threshold_from.read_text(encoding="utf-8"))["score_threshold"]
    else:
        threshold = args.threshold

    mean_seconds = float(np.mean(seconds)) if seconds else None
    summary = {
        "name": args.name, "predictions": str(args.predictions), "split": args.split, "target": args.target,
        "processed_images": len(rows), "ground_truth_objects": len(objects), "iou_threshold": IOU,
        "score_threshold": threshold, **at_threshold(detections, objects, threshold),
        "mean_detection_seconds": mean_seconds, "fps": 1 / mean_seconds if mean_seconds else None,
        "by_size": breakdown(detections, objects, threshold, "size"),
        "by_timeofday": breakdown(detections, objects, threshold, "timeofday"),
        "color": color_summary(objects, threshold) if args.target == "traffic_light" else None,
        "pr_curve": curve,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # dopasowane GT-światła z przewidzianym kolorem -> porównanie "na tych samych sygnalizatorach"
    matched = [{k: o[k] for k in ("image_id", "index", "color", "predicted_color")} for o in objects
               if o["matched_score"] is not None and o["matched_score"] >= threshold]
    (RUNS / "eval").mkdir(parents=True, exist_ok=True)
    (RUNS / "eval" / f"{args.name}_matched.json").write_text(json.dumps(matched), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("score_threshold", "precision_percent", "recall_percent",
                                              "f1_percent", "fps")}, indent=2))
    if summary["color"]:
        print(f"color accuracy: {summary['color']['color_accuracy_percent']:.2f}%")


if __name__ == "__main__":
    main()
