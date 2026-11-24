"""Wspólne ścieżki Lab B + dostęp (tylko do odczytu) do kodu Lab A."""
import sys
from pathlib import Path

LAB_B = Path(__file__).resolve().parents[1]
REPO = LAB_B.parent
LAB_A = REPO / "labA"

IMAGES = REPO / "bdd100k_images_100k" / "100k"   # {train,val,test}/*.jpg
LABELS = REPO / "bdd100k_labels" / "100k"        # {train,val,test}/*.json

CONFIGS = LAB_B / "configs"
RUNS = LAB_B / "runs"          # ciężkie artefakty (predykcje z niskim progiem, wagi) - gitignore
RESULTS = LAB_B / "results"    # małe podsumowania JSON/CSV do raportu - commitowane
ASSETS = LAB_B / "assets"      # figury do raportu

# Kod Lab A importujemy, nie kopiujemy: common.py (IoU, dopasowanie, metryki),
# traffic_light_colors.py (klasyczna metoda koloru HSV).
sys.path.insert(0, str(LAB_A / "code"))
