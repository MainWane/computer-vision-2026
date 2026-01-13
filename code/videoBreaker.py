import cv2
import os
from pathlib import Path

# Projektrod = computer-vision-2026
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# data/train/ugli
OUT_DIR = PROJECT_ROOT / "code" / "data" / "train" / "ugli"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture("C:/Users/ulrik/Desktop/cvmaterial/bv1.mp4")

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Gem hver 10. frame
    if frame_count % 10 == 0:
        out_path = OUT_DIR / f"frame_{saved_count:05d}.png"
        cv2.imwrite(str(out_path), frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Gemt {saved_count} frames ud af {frame_count}")