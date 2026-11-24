"""Wczytywanie obrazów i etykiet BDD100K (te same reguły co w Lab A)."""
import json
import random

from PIL import ImageFile

from paths import IMAGES, LABELS

# W BDD100K są ucięte JPEG-i (np. test/cba8da87-947f586b.jpg) - PIL wczytuje je zamiast rzucać OSError
ImageFile.LOAD_TRUNCATED_IMAGES = True
from common import valid_box  # Lab A

COLORS = ("red", "yellow", "green", "unknown")
CATEGORIES = {"traffic_sign": "traffic sign", "traffic_light": "traffic light"}


def split_images(split):
    return sorted((IMAGES / split).glob("*.jpg"))


def val_subset(count=1000, seed=42):
    """Podzbiór val do strojenia progów - ta sama procedura losowania co select_images() w Lab A."""
    return random.Random(seed).sample(split_images("val"), count)


def select(split, subset=None, limit=None):
    paths = val_subset() if subset == "val1000" else split_images(split)
    return paths[:limit] if limit else paths


def load_annotation(split, stem):
    """Boksy GT (xyxy) dla znaków i świateł, kolory świateł (kolejność = kolejność boksów), pora dnia."""
    data = json.loads((LABELS / split / f"{stem}.json").read_text(encoding="utf-8"))
    out = {
        "timeofday": (data.get("attributes") or {}).get("timeofday", "undefined"),
        "weather": (data.get("attributes") or {}).get("weather", "undefined"),
        "traffic_sign": [],
        "traffic_light": [],
        "light_colors": [],
    }
    for obj in data["frames"][0].get("objects", []):
        for target, category in CATEGORIES.items():
            if obj.get("category") != category or (box := valid_box(obj)) is None:
                continue
            out[target].append(box)
            if target == "traffic_light":
                color = (obj.get("attributes") or {}).get("trafficLightColor")
                out["light_colors"].append(color if color in COLORS[:3] else "unknown")
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row) + "\n")
