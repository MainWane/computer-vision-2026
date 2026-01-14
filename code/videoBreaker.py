import cv2
import os
import re
from pathlib import Path

# Projektrod = computer-vision-2026 
PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = "C:/Users/ulrik/Desktop/cvmaterial/utest.mp4"      
   # video
OUT_DIR = PROJECT_ROOT / "code" / "data" / "test" / "ugli"         # mappe hvor frames gemmes 
FRAME_STEP = 15                                                    # gemmer hver helst 30. frame, ca. 1 frame hver sekund 

os.makedirs(OUT_DIR, exist_ok=True)

# Udgangspunkt er højest eksisterende index
pattern = re.compile(r"frame_(\d+)\.(jpg|png)")

existing_indices = []

for fname in os.listdir(OUT_DIR):
    match = pattern.match(fname)
    if match:
        existing_indices.append(int(match.group(1)))

start_index = max(existing_indices) + 1 if existing_indices else 0
print(f"Starter frame-index fra: {start_index}")

# Åbner video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Kan ikke åbne video")

frame_count = 0
saved_count = 0
current_index = start_index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_STEP == 0:
        out_path = OUT_DIR / f"frame_{current_index:05d}.jpg"
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            print(f"FEJL: kunne ikke gemme {out_path}")
        current_index += 1
        saved_count += 1


    frame_count += 1

cap.release()

print(f"Færdig. Gemte {saved_count} frames.")


