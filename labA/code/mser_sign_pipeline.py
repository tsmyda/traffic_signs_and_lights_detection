import argparse
import csv
import json
import math
import random
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from color_features import color_histogram
from common import non_maximum_suppression
from edge_proposals import generate_edge_proposals
from common import fingerprint, ground_truth, intersection_over_union, match_predictions, metrics, read_json, select_images, write_json
from mser_proposals import expand_proposals, generate_proposals
from parallel_processing import parallel_map
from training_samples import load_training_samples


ROOT = Path(__file__).resolve().parents[1]
COLOR_PAYLOAD = {"metadata": {"color_features": {"grid": [2, 2], "bins": [12, 4, 4]}}}
EVALUATION_MODEL = None


def proposal_settings(args):
    settings = {
        "delta": args.delta,
        "min_area": args.min_area,
        "max_area_fraction": args.max_area_fraction,
        "max_variation": args.max_variation,
        "expansion_factors": sorted({1.0, *args.expansion_factors}),
    }
    if args.include_edges:
        settings["edge_settings"] = {
            "low_threshold": args.canny_low,
            "high_threshold": args.canny_high,
            "min_box_area": args.min_area,
            "max_area_fraction": args.max_area_fraction,
            "max_aspect_ratio": args.edge_max_aspect_ratio,
        }
    return settings


def proposals_for_image(image, settings):
    parameters = {key: value for key, value in settings.items() if key not in {"expansion_factors", "edge_settings"}}
    rgb = np.asarray(image)
    groups = [generate_proposals(rgb, **parameters)]
    if settings.get("edge_settings") is not None:
        groups.append(generate_edge_proposals(rgb, **settings["edge_settings"]))
    combined = {}
    for originals in groups:
        for factor in settings["expansion_factors"]:
            for proposal in expand_proposals(originals, image.size, factor):
                key = tuple(proposal["box"])
                if key not in combined:
                    combined[key] = proposal
                else:
                    combined[key]["sources"] = sorted(set(combined[key]["sources"]) | set(proposal["sources"]))
    return list(combined.values())


def describe(image, box, size):
    crop = image.crop(box).resize((size, size), Image.Resampling.BILINEAR)
    rgb = np.asarray(crop, dtype=np.float32) / 255
    edges = hog(rgb2gray(rgb), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", transform_sqrt=False)
    color = color_histogram(rgb, COLOR_PAYLOAD)
    width, height = box[2] - box[0], box[3] - box[1]
    geometry = np.asarray([math.log(width / height), width / image.width, height / image.height], dtype=np.float32)
    return np.concatenate((edges, color, geometry)).astype(np.float32)


def feature_count(size):
    return (size // 8 - 1) ** 2 * 36 + 83


def feature_settings(size):
    return {
        "window_size": size,
        "resize": "bilinear_stretch",
        "hog": {"orientations": 9, "pixels_per_cell": [8, 8], "cells_per_block": [2, 2], "block_norm": "L2-Hys", "transform_sqrt": False},
        "hsv": COLOR_PAYLOAD["metadata"]["color_features"],
        "geometry": ["log_aspect_ratio", "relative_width", "relative_height"],
    }


def prepare_image(task):
    path, samples_dir, settings = task
    output = samples_dir / f"{path.stem}.npz"
    report_path = samples_dir / f"{path.stem}.json"
    if output.exists() and report_path.exists():
        return path.name, read_json(report_path)
    target_class = settings.get("target_class", "traffic_sign")
    coverage_key = "covered_signs" if target_class == "traffic_sign" else "covered_lights"
    targets, invalid = ground_truth(path.with_suffix(".json"), target_class)
    with Image.open(path) as source:
        image = source.convert("RGB")
    proposals = proposals_for_image(image, settings["proposals"])
    overlaps = np.zeros((len(proposals), len(targets)), dtype=np.float32)
    if proposals and targets:
        boxes = np.asarray([row["box"] for row in proposals], dtype=np.float64)
        references = np.asarray(targets, dtype=np.float64)
        sizes = np.maximum(0, np.minimum(boxes[:, None, 2:], references[None, :, 2:]) - np.maximum(boxes[:, None, :2], references[None, :, :2]))
        intersection = sizes.prod(axis=2)
        union = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)[:, None] + (references[:, 2:] - references[:, :2]).prod(axis=1)[None, :] - intersection
        overlaps = (intersection / union).astype(np.float32)
    positive = set()
    covered = 0
    for target_index in range(len(targets)):
        indices = np.flatnonzero(overlaps[:, target_index] >= settings["positive_iou"])
        covered += int(len(indices) > 0)
        ordered = sorted(indices.tolist(), key=lambda item: (-float(overlaps[item, target_index]), item))
        positive.update(ordered[:settings["positives_per_sign"]])
    maximum = overlaps.max(axis=1) if targets else np.zeros(len(proposals))
    negative = np.flatnonzero(maximum < settings["negative_iou"]).tolist() if not invalid else []
    negative = random.Random(f"{settings['seed']}:{path.stem}").sample(negative, min(len(negative), settings["negatives_per_image"]))
    selected = sorted(positive) + negative
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.uint8)
    size = settings["features"]["window_size"]
    features = np.stack([describe(image, proposals[item]["box"], size) for item in selected]) if selected else np.empty((0, feature_count(size)), dtype=np.float32)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, features=features, labels=labels, boxes=np.asarray([proposals[item]["box"] for item in selected], dtype=np.int32).reshape(-1, 4))
    temporary.replace(output)
    report = {"image_path": str(path.resolve()), "positive_samples": len(positive), "negative_samples": len(negative), "ground_truth_objects": len(targets), coverage_key: covered, "invalid_ground_truth_objects": invalid, "total_proposals": len(proposals)}
    write_json(report_path, report)
    return path.name, report


def prepare(args):
    candidates = sorted((ROOT / "100k/train").glob("*.jpg"))
    if len(candidates) < args.images:
        raise ValueError("Not enough training images")
    images = random.Random(args.seed).sample(candidates, args.images)
    run = {
        "images": [str(path.resolve()) for path in images],
        "proposals": proposal_settings(args),
        "features": feature_settings(args.window_size),
        "positive_iou": 0.5,
        "negative_iou": 0.2,
        "positives_per_sign": args.positives_per_sign,
        "negatives_per_image": args.negatives_per_image,
        "seed": args.seed,
        "opencv_version": cv2.__version__,
    }
    if args.target_class != "traffic_sign":
        run["target_class"] = args.target_class
    coverage_key = "covered_signs" if args.target_class == "traffic_sign" else "covered_lights"
    settings_path = args.output_dir / "settings.json"
    if settings_path.exists():
        if read_json(settings_path) != run:
            raise ValueError("Preparation settings changed; choose another output directory")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is nonempty and has no preparation settings")
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings_path, run)
    counts = {"positive_samples": 0, "negative_samples": 0, "ground_truth_objects": 0, coverage_key: 0, "invalid_ground_truth_objects": 0}
    worker_settings = {key: value for key, value in run.items() if key != "images"}
    tasks = ((path, samples_dir, worker_settings) for path in images)
    for index, (name, report) in enumerate(parallel_map(prepare_image, tasks, args.workers), start=1):
        for key in counts:
            counts[key] += report[key]
        print(f"[{index}/{len(images)}] {name}: {report['positive_samples']} positive, {report['negative_samples']} negative", flush=True)
        write_json(args.output_dir / "summary.json", {"processed_images": index, "complete": index == len(images), **counts})
    print(json.dumps(counts, indent=2))


def train(args):
    settings = read_json(args.dataset_dir / "settings.json")
    summary = read_json(args.dataset_dir / "summary.json")
    if not summary["complete"]:
        raise ValueError("Finish dataset preparation before training")
    if args.output_dir.exists() and any(path.name not in {"sampling.json", "mser_random_forest.joblib.tmp"} for path in args.output_dir.iterdir()):
        raise ValueError("Model directory exists; choose another output directory")
    features, labels, sampling = load_training_samples(args.dataset_dir, settings, feature_count(settings["features"]["window_size"]), args.max_positive_samples, args.max_negative_samples, args.sample_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "sampling.json", sampling)
    print(json.dumps(sampling), flush=True)
    if args.classifier == "random-forest":
        start = time.perf_counter()
        model = RandomForestClassifier(
            n_estimators=args.trees,
            min_samples_leaf=args.min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=args.jobs,
            random_state=42,
        )
        model.fit(features, labels)
        report = {
            "classifier": "random-forest",
            "trees": args.trees,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "fit_seconds": time.perf_counter() - start,
            "positive_samples": int(labels.sum()),
            "negative_samples": int((labels == 0).sum()),
            "sampling": sampling,
        }
        payload = {"model": model, "settings": settings, "training": report, "feature_version": 1, "classifier": "random-forest", "score_type": "positive_class_probability"}
        temporary = args.output_dir / "mser_random_forest.joblib.tmp"
        joblib.dump(payload, temporary)
        temporary.replace(args.output_dir / "mser_random_forest.joblib")
        write_json(args.output_dir / "training.json", [report])
        print(json.dumps(report), flush=True)
        return
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    del features
    reports = []
    for c_value in args.c_values:
        start = time.perf_counter()
        classifier = LinearSVC(C=c_value, class_weight="balanced", dual="auto", max_iter=args.max_iter, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(scaled, labels)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        model = make_pipeline(scaler, classifier)
        report = {"c_value": c_value, "converged": converged, "iterations": int(classifier.n_iter_), "fit_seconds": time.perf_counter() - start, "positive_samples": int(labels.sum()), "negative_samples": int((labels == 0).sum())}
        payload = {"model": model, "settings": settings, "training": report, "feature_version": 1}
        joblib.dump(payload, args.output_dir / f"mser_svm_C_{c_value:g}.joblib")
        reports.append(report)
        write_json(args.output_dir / "training.json", reports)
        print(json.dumps(report), flush=True)


def predict(image, payload, threshold, nms_iou):
    start = time.perf_counter()
    proposals = proposals_for_image(image, payload["settings"]["proposals"])
    candidates = []
    size = payload["settings"]["features"]["window_size"]
    target_class = payload["settings"].get("target_class", "traffic_sign")
    for offset in range(0, len(proposals), 256):
        batch = proposals[offset:offset + 256]
        features = np.stack([describe(image, row["box"], size) for row in batch])
        if payload.get("classifier", "svm") == "random-forest":
            scores = payload["model"].predict_proba(features)[:, 1]
        else:
            scores = payload["model"].decision_function(features)
        candidates.extend({"class": target_class, "box": row["box"], "score": float(score)} for row, score in zip(batch, scores) if score >= threshold)
    predictions = non_maximum_suppression(candidates, nms_iou)
    return {"predictions": predictions, "proposal_count": len(proposals), "accepted_before_nms": len(candidates), "detection_seconds": time.perf_counter() - start}


def load_model(path):
    payload = joblib.load(path)
    if payload.get("feature_version") != 1 or list(payload["model"].classes_) != [0, 1]:
        raise ValueError("Unexpected MSER model format")
    if payload["settings"].get("target_class", "traffic_sign") not in {"traffic_sign", "traffic_light"}:
        raise ValueError("Unsupported detector target class")
    if payload.get("classifier", "svm") == "random-forest":
        payload["model"].n_jobs = 1
    elif not payload["training"]["converged"]:
        raise ValueError("Model did not converge; train again with a higher --max-iter")
    return payload


def visualize(image, predictions, path):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for row in predictions:
        draw.rectangle(row["box"], outline="cyan", width=2)
    preview.save(path, quality=95)


def detect(args):
    path = args.image.resolve()
    if path.parent not in {(ROOT / "100k/train").resolve(), (ROOT / "100k/val").resolve()}:
        raise ValueError("Use a training or validation image during development")
    payload = load_model(args.model)
    if args.output_dir.exists():
        raise ValueError("Output directory exists; choose another output directory")
    with Image.open(path) as source:
        image = source.convert("RGB")
    report = predict(image, payload, args.score_threshold, args.nms_iou)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "predictions.json", {"image_path": str(path), "model_sha256": fingerprint(args.model), "score_threshold": args.score_threshold, "nms_iou": args.nms_iou, **report})
    visualize(image, report["predictions"], args.output_dir / "detections.jpg")
    print(f"Saved {len(report['predictions'])} detections from {report['proposal_count']} proposals in {report['detection_seconds']:.2f}s")


def initialize_evaluation(model_path):
    global EVALUATION_MODEL
    EVALUATION_MODEL = load_model(model_path)


def evaluate_image(task):
    path, directory, threshold, nms_iou, save_preview = task
    target_class = EVALUATION_MODEL["settings"].get("target_class", "traffic_sign")
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "predictions.json"
    if report_path.exists():
        report = read_json(report_path)
    else:
        with Image.open(path) as source:
            image = source.convert("RGB")
        report = predict(image, EVALUATION_MODEL, threshold, nms_iou)
        if save_preview:
            visualize(image, report["predictions"], directory / "detections.jpg")
        write_json(report_path, report)
    targets, invalid = ground_truth(path.with_suffix(".json"), target_class)
    matching = match_predictions(report["predictions"], targets, 0.5)
    write_json(directory / "matching.json", matching)
    return {"image_id": path.stem, "ground_truth_objects": len(targets), "invalid_ground_truth_objects": invalid, "predicted_objects": len(report["predictions"]), "proposal_count": report["proposal_count"], "detection_seconds": report["detection_seconds"], **metrics(matching["true_positive"], matching["false_positive"], matching["false_negative"])}


def evaluate(args):
    if args.split == "test" and args.score_threshold_file is None:
        raise ValueError("Test evaluation requires a frozen validation --score-threshold-file")
    if args.score_threshold_file is not None:
        frozen = read_json(args.score_threshold_file)
        if frozen.get("source_split") not in {"validation_subset", "val"}:
            raise ValueError("Threshold file must come from validation")
        if frozen.get("model_sha256") != fingerprint(args.model):
            raise ValueError("Threshold file belongs to a different model")
        if frozen.get("nms_iou") != args.nms_iou:
            raise ValueError("NMS IoU differs from the validation threshold file")
        args.score_threshold = frozen["best_validation_threshold"]["score_threshold"]
        if not math.isfinite(args.score_threshold):
            raise ValueError("Frozen score threshold must be finite")
    split_dir = ROOT / "100k" / args.split
    image_list = args.image_list
    if image_list is None:
        image_list = ROOT / "results/sign_diagnostics_hard_negatives_v1/settings.json" if args.split == "val" else args.output_dir / "unused_image_list.json"
    count = args.images
    if args.all_images:
        count = len(list(split_dir.glob("*.jpg")))
        image_list = args.output_dir / "unused_image_list.json"
    if count < 1:
        raise ValueError(f"No images found in {split_dir}")
    images = select_images(SimpleNamespace(val_dir=split_dir, image_list=image_list, images=count, seed=args.seed))
    payload = load_model(args.model)
    target_class = payload["settings"].get("target_class", "traffic_sign")
    run = {"images": [str(path.resolve()) for path in images], "model_sha256": fingerprint(args.model), "annotation_hashes": {path.stem: fingerprint(path.with_suffix(".json")) for path in images}, "score_threshold": args.score_threshold, "nms_iou": args.nms_iou, "evaluation_iou": 0.5, "opencv_version": cv2.__version__}
    if target_class != "traffic_sign":
        run["class"] = target_class
    if args.split != "val":
        run["split"] = args.split
        run["threshold_file_sha256"] = fingerprint(args.score_threshold_file)
    settings_path = args.output_dir / "settings.json"
    if settings_path.exists():
        if read_json(settings_path) != run:
            raise ValueError("Evaluation settings changed; choose another output directory")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is nonempty and has no evaluation settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(settings_path, run)
    del payload
    rows = []
    totals = {key: 0 for key in ("ground_truth_objects", "predicted_objects", "invalid_ground_truth_objects", "true_positive", "false_positive", "false_negative", "detection_seconds")}
    tasks = ((path, args.output_dir / "images" / path.stem, args.score_threshold, args.nms_iou, index < args.preview_images) for index, path in enumerate(images))
    csv_path = args.output_dir / "per_image.csv"
    for index, row in enumerate(parallel_map(evaluate_image, tasks, args.workers, initialize_evaluation, (args.model,)), start=1):
        rows.append(row)
        for key in totals:
            totals[key] += row[key]
        summary = {
            "class": target_class, "split": "validation_subset" if args.split == "val" else "test", "processed_images": index, "requested_images": len(images), "complete": index == len(images), "ground_truth_objects": totals["ground_truth_objects"], "predicted_objects": totals["predicted_objects"], "invalid_ground_truth_objects": totals["invalid_ground_truth_objects"], "iou_threshold": 0.5, "score_threshold": args.score_threshold,
            **metrics(*(totals[key] for key in ("true_positive", "false_positive", "false_negative"))),
            "mean_detection_seconds": totals["detection_seconds"] / index,
            "note": "One-to-one matching after NMS. Test results use a frozen validation threshold." if args.split == "test" else "One-to-one matching after NMS on validation data; not final test results.",
        }
        write_json(args.output_dir / "summary.json", summary)
        with csv_path.open("w" if index == 1 else "a", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            if index == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f"[{index}/{len(images)}] P={summary['precision_percent']:.3f}% R={summary['recall_percent']:.3f}% F1={summary['f1_percent']:.3f}%", flush=True)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare, train and evaluate classical traffic sign or light detectors using MSER, optional Canny, HOG and HSV.")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--target-class", choices=["traffic_sign", "traffic_light"], default="traffic_sign")
    preparation.add_argument("--images", type=int, default=2000)
    preparation.add_argument("--workers", type=int, default=1)
    preparation.add_argument("--seed", type=int, default=42)
    preparation.add_argument("--window-size", type=int, default=32)
    preparation.add_argument("--positives-per-sign", "--positives-per-object", dest="positives_per_sign", type=int, default=5)
    preparation.add_argument("--negatives-per-image", type=int, default=100)
    preparation.add_argument("--delta", type=int, default=5)
    preparation.add_argument("--min-area", type=int, default=16)
    preparation.add_argument("--max-area-fraction", type=float, default=0.25)
    preparation.add_argument("--max-variation", type=float, default=0.25)
    preparation.add_argument("--expansion-factors", type=float, nargs="+", default=[1, 1.25, 1.5, 2])
    preparation.add_argument("--include-edges", action="store_true")
    preparation.add_argument("--canny-low", type=int, default=50)
    preparation.add_argument("--canny-high", type=int, default=150)
    preparation.add_argument("--edge-max-aspect-ratio", type=float, default=10)
    preparation.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/mser_signs_v1")
    training = commands.add_parser("train")
    training.add_argument("--dataset-dir", type=Path, default=ROOT / "data/processed/mser_signs_v1")
    training.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/mser_signs_v1")
    training.add_argument("--c-values", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    training.add_argument("--max-iter", type=int, default=20000)
    training.add_argument("--classifier", choices=["svm", "random-forest"], default="svm")
    training.add_argument("--trees", type=int, default=200)
    training.add_argument("--min-samples-leaf", type=int, default=2)
    training.add_argument("--jobs", type=int, default=-1)
    training.add_argument("--max-positive-samples", type=int)
    training.add_argument("--max-negative-samples", type=int)
    training.add_argument("--sample-seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--images", type=int, default=20)
    evaluation.add_argument("--all-images", action="store_true")
    evaluation.add_argument("--split", choices=["val", "test"], default="val")
    evaluation.add_argument("--workers", type=int, default=1)
    evaluation.add_argument("--preview-images", type=int, default=20)
    evaluation.add_argument("--score-threshold-file", type=Path)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--image-list", type=Path)
    evaluation.set_defaults(val_dir=ROOT / "100k/val")
    detection = commands.add_parser("detect")
    detection.add_argument("--image", type=Path, required=True)
    for command in (evaluation, detection):
        command.add_argument("--model", type=Path, default=ROOT / "checkpoints/mser_signs_v1/mser_svm_C_0.01.joblib")
        command.add_argument("--score-threshold", type=float, default=-2)
        command.add_argument("--nms-iou", type=float, default=0.3)
    detection.set_defaults(score_threshold=0)
    evaluation.add_argument("--output-dir", type=Path, default=ROOT / "results/mser_sign_evaluation_v1")
    detection.add_argument("--output-dir", type=Path, default=ROOT / "results/mser_sign_detection_v1")
    args = parser.parse_args()
    if args.command in {"prepare", "evaluate"} and args.workers < 1:
        parser.error("Worker count must be positive")
    if args.command == "prepare":
        if not 0 <= args.canny_low < args.canny_high <= 255:
            parser.error("Canny thresholds must satisfy 0 <= low < high <= 255")
        if not math.isfinite(args.edge_max_aspect_ratio) or args.edge_max_aspect_ratio < 1:
            parser.error("Maximum edge aspect ratio must be finite and at least 1")
        if min(args.images, args.positives_per_sign, args.negatives_per_image, args.delta, args.min_area) < 1:
            parser.error("Counts and delta must be positive")
        if args.window_size < 16 or args.window_size % 8:
            parser.error("Window size must be a multiple of 8 and at least 16")
        if not all(math.isfinite(value) and 0 < value <= 1 for value in (args.max_area_fraction, args.max_variation)):
            parser.error("Area fraction and variation must be in (0, 1]")
        if not all(math.isfinite(value) and value >= 1 for value in args.expansion_factors):
            parser.error("Expansion factors must be finite and at least 1")
    if args.command == "train":
        if any(value is not None and value < 1 for value in (args.max_positive_samples, args.max_negative_samples)):
            parser.error("Training sample limits must be positive")
        if min(args.trees, args.min_samples_leaf) < 1 or args.jobs == 0:
            parser.error("Tree count and minimum leaf size must be positive; jobs cannot be zero")
        if args.max_iter < 1 or not all(math.isfinite(value) and value > 0 for value in args.c_values):
            parser.error("C values and maximum iterations must be positive")
    if args.command in {"detect", "evaluate"}:
        if not math.isfinite(args.score_threshold) or not 0 <= args.nms_iou <= 1:
            parser.error("Invalid score threshold or NMS IoU")
    if args.command == "evaluate" and args.images < 1:
        parser.error("Image count must be positive")
    if args.command == "evaluate" and args.preview_images < 0:
        parser.error("Preview image count cannot be negative")
    {"prepare": prepare, "train": train, "evaluate": evaluate, "detect": detect}[args.command](args)


if __name__ == "__main__":
    main()
