"""Inferencja YOLO-COCO -> JSONL w formacie predykcji z Lab A.

Zapisujemy wszystkie detekcje od niskiego progu `conf` (np. 0.01), a właściwy próg wybieramy
później na val w evaluate.py (sweep). Czas mierzony per obraz, end-to-end (odczyt + sieć + NMS).

Przykład:
    python labB/code/predict.py --config labB/configs/yolo_coco.json --split val --subset val1000 --imgsz 1280
    python labB/code/predict.py --config labB/configs/yolo_coco.json --split test
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from paths import RUNS
from bdd import select

LOGGER.setLevel("ERROR")  # bez setek "'half' is deprecated" na każdy obraz


def read_image(path):
    """Odczyt BGR przez OpenCV. W BDD100K są ucięte JPEG-i (np. test/cba8da87-947f586b.jpg):
    loader Ultralytics zwraca dla nich pustą listę wyników, a OpenCV dekoduje je (z brakującym końcem)."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def run(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    weights = args.weights or config["weights"]
    imgsz = args.imgsz or config["imgsz"]
    class_map = {int(key): value for key, value in config["class_map"].items()}
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    half = config.get("half", True) and device != "cpu"

    name = args.name or f"{config['name']}_{imgsz}_{args.subset or args.split}" + ("_cpu" if device == "cpu" else "")
    output = RUNS / "predictions" / f"{name}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():  # wznawianie przerwanego przebiegu
        with open(output, encoding="utf-8") as source:
            done = {json.loads(line)["image_id"] for line in source if line.strip()}

    model = YOLO(weights)
    options = dict(imgsz=imgsz, conf=config["conf"], iou=config["iou"], classes=sorted(class_map),
                   max_det=config["max_det"], agnostic_nms=config.get("agnostic_nms", False),
                   device=device, half=half, verbose=False)
    images = select(args.split, args.subset, args.limit)
    for _ in range(10):  # rozgrzewka (cuDNN autotune, alokacje) - nie wchodzi do pomiaru
        model.predict(read_image(images[0]), **options)

    with open(output, "a", encoding="utf-8") as target:
        for index, path in enumerate(images, 1):
            if path.stem in done:
                continue
            start = time.perf_counter()
            image = read_image(path)
            results = model.predict(image, **options) if image is not None else []
            if results:
                boxes = results[0].boxes
                xyxy, scores, classes = boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.int().tolist()
            else:  # nieczytelny obraz: brak detekcji (jego GT liczą się jako FN, jak w Lab A)
                print(f"WARNING: nie można odczytać {path.name}", flush=True)
                xyxy, scores, classes = [], [], []
            if device != "cpu":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            predictions = [{"class": class_map[c], "box": [round(v, 2) for v in box], "score": s}
                           for box, s, c in zip(xyxy, scores, classes)]
            target.write(json.dumps({"image_id": path.stem, "image_filename": path.name,
                                     "seconds": seconds, "predictions": predictions}) + "\n")
            if index % 500 == 0:
                print(f"[{index}/{len(images)}] {name}", flush=True)

    meta = {"config": config, "weights": str(weights), "imgsz": imgsz, "device": device, "half": half,
            "split": args.split, "subset": args.subset, "images": len(images),
            "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None}
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--subset", choices=["val1000"], help="podzbiór do strojenia (jak w Lab A)")
    parser.add_argument("--limit", type=int, help="tylko pierwsze N obrazów (np. pomiar FPS na CPU)")
    parser.add_argument("--weights", help="nadpisuje config['weights']")
    parser.add_argument("--imgsz", type=int, help="nadpisuje config['imgsz']")
    parser.add_argument("--device", default="0", help="'0' = pierwsza karta CUDA, 'cpu' = procesor")
    parser.add_argument("--name", help="nazwa pliku wyjściowego (domyślnie z configu)")
    args = parser.parse_args()
    if args.subset and args.split != "val":
        parser.error("--subset val1000 wymaga --split val")
    run(args)


if __name__ == "__main__":
    main()
