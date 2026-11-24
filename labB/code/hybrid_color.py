"""Zad. 2 - podejście hybrydowe: boksy świateł z YOLO -> kolor klasyczną metodą HSV z Lab A.

Wariant H0: niezmieniony labA/configs/traffic_light_colors.json (różni się tylko detektor).
Wariant H1: --color-config labB/configs/traffic_light_colors_yolo.json (z tune_hsv.py, strojony na val).
Tryb --gt-boxes: kolor liczony na boksach GT (oracle, górna granica dla samej metody koloru).

Przykład:
    python labB/code/hybrid_color.py --predictions labB/runs/predictions/yolo_coco_1280_test.jsonl \
        --min-score 0.25 --split test --name hybrid_H0_test
    python labB/code/hybrid_color.py --gt-boxes --split test --name oracle_H0_test
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from paths import IMAGES, LAB_A, RUNS
from bdd import load_annotation, read_jsonl, split_images, write_jsonl
from traffic_light_colors import classify_color, validate_config  # Lab A


def is_light(name):
    return name.startswith("traffic_light")


def shrink(box, fraction):
    """Obcięcie ROI z każdej strony o ułamek boku (parametr H1 'roi_shrink'; 0 = jak w Lab A)."""
    dx, dy = (box[2] - box[0]) * fraction, (box[3] - box[1]) * fraction
    return [box[0] + dx, box[1] + dy, box[2] - dx, box[3] - dy]


def colorize(rows, split, config, min_score):
    for row in rows:
        lights = [p for p in row["predictions"] if is_light(p["class"]) and p["score"] >= min_score]
        row["predictions"], row["color_seconds"] = lights, 0.0
        if not lights:
            yield row
            continue
        with Image.open(IMAGES / split / row["image_filename"]) as source:
            image = source.convert("RGB")
        # mierzymy tylko klasyfikację koloru - obraz w systemie end-to-end jest już w pamięci po detekcji
        start = time.perf_counter()
        for prediction in lights:
            result = classify_color(image, shrink(prediction["box"], config.get("roi_shrink", 0.0)), config)
            prediction.update(result, **{"class": f"traffic_light_{result['color']}"})
        row["color_seconds"] = time.perf_counter() - start
        yield row


def gt_rows(split):
    for path in split_images(split):
        annotation = load_annotation(split, path.stem)
        yield {"image_id": path.stem, "image_filename": path.name, "seconds": 0.0,
               "predictions": [{"class": "traffic_light", "box": box, "score": 1.0}
                               for box in annotation["traffic_light"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, help="JSONL z predict.py")
    parser.add_argument("--gt-boxes", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--color-config", type=Path, default=LAB_A / "configs/traffic_light_colors.json")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="koloruj tylko detekcje >= progu (zamrożony próg z val)")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.gt_boxes == (args.predictions is not None):
        parser.error("Podaj dokładnie jedno: --predictions albo --gt-boxes")

    config = json.loads(args.color_config.read_text(encoding="utf-8"))
    validate_config(config)
    rows = gt_rows(args.split) if args.gt_boxes else read_jsonl(args.predictions)
    output = RUNS / "predictions" / f"{args.name}.jsonl"
    write_jsonl(output, colorize(rows, args.split, config, args.min_score))
    source_meta = args.predictions.with_suffix(".meta.json") if args.predictions else None
    gpu = json.loads(source_meta.read_text(encoding="utf-8")).get("gpu") if source_meta and source_meta.exists() else None
    output.with_suffix(".meta.json").write_text(json.dumps({
        "gpu": gpu, "source": "gt_boxes" if args.gt_boxes else str(args.predictions),
        "color_config_path": str(args.color_config), "color_config": config,
        "min_score": args.min_score, "split": args.split}, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
