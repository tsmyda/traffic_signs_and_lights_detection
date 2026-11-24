import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import fingerprint, read_json, write_json
from common import valid_box
from traffic_light_colors import COLORS, classify_color, validate_config


ROOT = Path(__file__).resolve().parents[1]


def target_colors(path):
    annotation = read_json(path)
    frames = annotation.get("frames", [])
    if len(frames) != 1:
        raise ValueError(f"Expected one annotation frame: {path}")
    colors = []
    for obj in frames[0].get("objects", []):
        if obj.get("category") != "traffic light" or valid_box(obj) is None:
            continue
        color = (obj.get("attributes") or {}).get("trafficLightColor")
        colors.append(color if color in COLORS[:3] else "unknown")
    return colors


def save_preview(image, predictions, path):
    preview = Image.new("RGB", (image.width, image.height + 32), "#202020")
    preview.paste(image, (0, 32))
    draw = ImageDraw.Draw(preview)
    display_colors = {"red": "red", "yellow": "yellow", "green": "lime", "unknown": "white"}
    for index, color in enumerate(COLORS):
        draw.text((10 + index * 120, 10), color, fill=display_colors[color])
    for row in predictions:
        x1, y1, x2, y2 = row["box"]
        draw.rectangle((x1, y1 + 32, x2, y2 + 32), outline=display_colors[row["color"]], width=2)
    preview.save(path, quality=95)


def evaluate_colors(args):
    evaluation = read_json(args.evaluation_dir / "summary.json")
    settings = read_json(args.evaluation_dir / "settings.json")
    if not evaluation["complete"] or evaluation.get("class") != "traffic_light":
        raise ValueError("A completed traffic_light evaluation is required")
    expected_split = "test" if args.split == "test" else "validation_subset"
    if evaluation.get("split") != expected_split:
        raise ValueError("Color split must match the detection evaluation split")
    config = read_json(args.config)
    validate_config(config)
    threshold = args.score_threshold
    if threshold is None:
        if args.split == "test":
            if "threshold_file_sha256" not in settings:
                raise ValueError("Test detections must use a frozen validation threshold")
            threshold = settings["score_threshold"]
        else:
            selected = read_json(args.evaluation_dir / "threshold_analysis/best_threshold.json")
            threshold = selected["best_validation_threshold"]["score_threshold"]
    if args.split == "test" and threshold != settings["score_threshold"]:
        raise ValueError("Test colors must use the frozen detection threshold")
    if not math.isfinite(threshold) or threshold < settings["score_threshold"]:
        raise ValueError("Color threshold must be finite and at least the saved detection threshold")
    images = [Path(value) for value in settings["images"]]
    if not images or any(path.resolve().parent != (ROOT / "100k" / args.split).resolve() for path in images):
        raise ValueError("Images must belong to the selected split")
    run = {"evaluation_dir": str(args.evaluation_dir.resolve()), "evaluation_settings_sha256": fingerprint(args.evaluation_dir / "settings.json"), "score_threshold": threshold, "color_config": config, "color_evaluation": "true_positive_detections_only", "split": args.split}
    settings_path = args.output_dir / "settings.json"
    if settings_path.exists():
        if read_json(settings_path) != run:
            raise ValueError("Color evaluation settings changed; choose another output directory")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Color output directory is nonempty and has no settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings_path, run)
    confusion = np.zeros((4, 4), dtype=np.int64)
    rows = []
    total_predictions = 0
    for index, path in enumerate(images, start=1):
        annotation_path = path.with_suffix(".json")
        expected_hash = settings.get("annotation_hashes", {}).get(path.stem)
        if expected_hash is not None and fingerprint(annotation_path) != expected_hash:
            raise ValueError(f"Annotation changed since detection evaluation: {path}")
        directory = args.output_dir / "images" / path.stem
        cached_path = directory / "predictions.json"
        if cached_path.exists():
            cached = read_json(cached_path)
            if "evaluation_rows" in cached and "confusion_matrix" in cached:
                rows.extend(cached["evaluation_rows"])
                confusion += np.asarray(cached["confusion_matrix"], dtype=np.int64)
                total_predictions += len(cached["predictions"])
                print(f"[{index}/{len(images)}] {path.name}: cached colors", flush=True)
                continue
        matching = read_json(args.evaluation_dir / "images" / path.stem / "matching.json")
        colors = target_colors(annotation_path)
        with Image.open(path) as source:
            image = source.convert("RGB")
        predictions = []
        image_rows = []
        image_confusion = np.zeros((4, 4), dtype=np.int64)
        for prediction_index, prediction in enumerate(matching["matched_predictions"]):
            if prediction["score"] < threshold:
                continue
            if prediction.get("class") != "traffic_light":
                raise ValueError("Unexpected prediction class")
            classified = classify_color(image, prediction["box"], config)
            actual = None
            if prediction["result"] == "true_positive":
                actual = colors[prediction["matched_target_index"]]
                image_confusion[COLORS.index(actual), COLORS.index(classified["color"])] += 1
            predictions.append({**prediction, **classified, "class": f"traffic_light_{classified['color']}"})
            image_rows.append({"image_id": path.stem, "prediction_index": prediction_index, "score": prediction["score"], "detection_result": prediction["result"], "predicted_color": classified["color"], "target_color": actual, "color_correct": classified["color"] == actual if actual is not None else None})
        rows.extend(image_rows)
        confusion += image_confusion
        directory.mkdir(parents=True, exist_ok=True)
        if index <= args.preview_images:
            save_preview(image, predictions, directory / "colors.jpg")
        write_json(directory / "predictions.json", {"image_path": str(path.resolve()), "predictions": predictions, "evaluation_rows": image_rows, "confusion_matrix": image_confusion.tolist()})
        total_predictions += len(predictions)
        print(f"[{index}/{len(images)}] {path.name}: {len(predictions)} predictions", flush=True)
    per_class = []
    for index, color in enumerate(COLORS):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        per_class.append({"color": color, "support": support, "precision": tp / predicted if predicted else 0, "recall": tp / support if support else 0, "f1": 2 * tp / (support + predicted) if support + predicted else 0})
    evaluated = int(confusion.sum())
    summary = {
        "split": args.split,
        "complete": True,
        "processed_images": len(images),
        "score_threshold": threshold,
        "predicted_objects": total_predictions,
        "correctly_localized_lights": evaluated,
        "correct_colors": int(confusion.trace()),
        "color_accuracy_percent": 100 * int(confusion.trace()) / evaluated if evaluated else None,
        "per_class": per_class,
        "confusion_labels": list(COLORS),
        "confusion_matrix": confusion.tolist(),
        "note": "Color metrics include only one-to-one matched detections at the chosen threshold. Matrix rows are reference colors; columns are predictions. Unknown reference colors mean unspecified annotations, not necessarily unlit lamps. HSV parameters are initial settings; scores are not probabilities. Per-class undefined precision, recall and F1 are reported as zero.",
    }
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["image_id", "prediction_index", "score", "detection_result", "predicted_color", "target_color", "color_correct"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["reference_color", *COLORS])
        writer.writerows([color, *confusion[index].tolist()] for index, color in enumerate(COLORS))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Classify traffic light colors in saved validation detections and evaluate matched lights only.")
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "results/traffic_light_rf_evaluation_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/traffic_light_colors.json")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--preview-images", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/traffic_light_colors_v1")
    args = parser.parse_args()
    if args.preview_images < 0:
        parser.error("Preview image count cannot be negative")
    evaluate_colors(args)


if __name__ == "__main__":
    main()
