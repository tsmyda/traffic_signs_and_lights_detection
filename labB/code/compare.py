"""Zad. 4: tabela porównawcza Lab A vs Lab B na teście + kolor na wspólnych sygnalizatorach.

Wymaga wcześniejszego evaluate.py dla każdej metody (pliki labB/results/<name>.json).
Lab A: liczby z labA/assets/test_results.json (FPS z oryginalnego pomiaru CPU) i - dla koloru na
części wspólnej - evaluate.py uruchomiony na labA/predictions/traffic_light_colors.jsonl (name=labA_light_test).

    python labB/code/compare.py --methods "hybrid_H0_test=YOLO-COCO + HSV (H0)" \
        "hybrid_H1_test=YOLO-COCO + HSV (H1)" "oracle_H0_test=GT boksy + HSV (oracle H0)"
"""
import argparse
import csv
import json
from pathlib import Path

from paths import LAB_A, RESULTS, RUNS

COLUMNS = ["method", "class", "precision_percent", "recall_percent", "f1_percent",
           "color_accuracy_percent", "mean_detection_seconds", "fps", "hardware"]


def lab_a_rows():
    data = json.loads((LAB_A / "assets/test_results.json").read_text(encoding="utf-8"))
    rows = []
    for entry in data["detection"]:
        rows.append({"method": "Lab A: MSER+Canny+RF" + (" + HSV" if entry["class"] == "traffic_light" else ""),
                     "class": entry["class"], **{k: entry[k] for k in COLUMNS[2:5]},
                     "color_accuracy_percent": data["color"]["color_accuracy_percent"] if entry["class"] == "traffic_light" else None,
                     "mean_detection_seconds": entry["mean_detection_seconds"],
                     "fps": 1 / entry["mean_detection_seconds"], "hardware": "CPU"})
    return rows


def matched(name):
    path = RUNS / "eval" / f"{name}_matched.json"
    return {(m["image_id"], m["index"]): m for m in json.loads(path.read_text())} if path.exists() else None


def common_accuracy(first, second):
    keys = first.keys() & second.keys()
    if not keys:
        return None
    acc = lambda d: 100 * sum(d[k]["color"] == d[k]["predicted_color"] for k in keys) / len(keys)
    return {"common_lights": len(keys), "first_accuracy_percent": acc(first), "second_accuracy_percent": acc(second)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", required=True, help="name=etykieta (name = plik results/<name>.json)")
    parser.add_argument("--lab-a-matched", default="labA_light_test")
    args = parser.parse_args()

    rows = lab_a_rows()
    reference = matched(args.lab_a_matched)
    common = {}
    for item in args.methods:
        name, label = item.split("=", 1)
        summary = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
        meta_path = Path(summary["predictions"]).with_suffix(".meta.json")
        hardware = "?"
        if meta_path.exists():
            hardware = json.loads(meta_path.read_text(encoding="utf-8")).get("gpu") or "CPU"
        rows.append({"method": f"Lab B: {label}", "class": summary["target"],
                     **{k: summary[k] for k in COLUMNS[2:5]},
                     "color_accuracy_percent": (summary["color"] or {}).get("color_accuracy_percent"),
                     "mean_detection_seconds": summary["mean_detection_seconds"], "fps": summary["fps"],
                     "hardware": hardware})
        if summary["target"] == "traffic_light" and reference and (other := matched(name)):
            common[label] = common_accuracy(reference, other)

    with open(RESULTS / "comparison.csv", "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "comparison_common_lights.json").write_text(json.dumps(common, indent=2), encoding="utf-8")

    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else ("-" if v is None else str(v))
    print("| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS))
    for row in rows:
        print("| " + " | ".join(fmt(row[c]) for c in COLUMNS) + " |")
    print("\nKolor na wspólnych sygnalizatorach (Lab A vs metoda):", json.dumps(common, indent=2))


if __name__ == "__main__":
    main()
