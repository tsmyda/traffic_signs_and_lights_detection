* **Wybrany YOLO-COCO:** coco_yolo11s_1280 (best F1 na val1000 = 42.67 %, próg 0.147).
* **Światła (test):** Lab A F1 21.40 % (P 20.6, R 22.3) -> YOLO-COCO F1 39.37 % (P 40.3, R 38.4).
* **Kolor (TP):** Lab A 83.98 %, hybryda H0 69.60 %, H1 72.94 %, oracle H0 65.58 %, oracle H1 65.39 %.
* **Szybkość:** Lab A 0.29 FPS (CPU); YOLO-COCO + HSV 38.3 FPS (GPU), 6.39 FPS (CPU).
* **STOP:** 3464 detekcji, precyzja 59.1 %, pokrycie znaków BDD 2.97 %.
* **Małe światła 0-8 px:** recall Lab A 13.6 %, YOLO-COCO 13.1 %.
* **Małe światła 8-16 px:** recall Lab A 26.2 %, YOLO-COCO 35.4 %.
