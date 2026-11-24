# Lab 04 — wykrywanie znaków i sygnalizatorów na BDD100K

```
bdd100k_images_100k/100k/{train,val,test}/*.jpg   # dane (gitignore), wspólne dla obu labów
bdd100k_labels/100k/{train,val,test}/*.json
labA/          # Lab A: metody klasyczne (MSER+Canny, HOG/HSV, Random Forest) - zamknięte, tylko do odczytu
  code/ configs/ assets/ predictions/ prototype/ report.{ipynb,html,pdf}
labB/          # Lab B: YOLO (COCO) + hybryda YOLO + HSV (bez fine-tuningu)
  PLAN.md      # plan i kolejność uruchamiania
  code/        # skrypty (importują labA/code/common.py i traffic_light_colors.py)
  configs/     # konfiguracje inferencji / treningu
  results/     # małe podsumowania JSON/CSV (commitowane)
  assets/      # figury do raportu
  runs/ data/  # ciężkie artefakty: predykcje, wagi, dataset YOLO (gitignore)
```

## Środowisko

```bash
uv sync                 # Lab A (CPU)
uv sync --group labb    # + Lab B: ultralytics, torch CUDA 12.8
```

Wszystkie skrypty uruchamiamy z katalogu głównego repo, np. `uv run python labB/code/predict.py ...`.
Raport Lab A: `cd labA && jupyter nbconvert --execute report.ipynb` (ścieżki względne do `labA/`).
