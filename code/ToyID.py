import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from pathlib import Path

CAMERA_INDEX = 0
TARGET_WIDTH = 640

# CNN
MODEL_PATH = "bamsenet_resnet182.pt"
BAMSE_CLASSES = ["NotBamse", "panda", "ugli"]
BAMSE_TARGETS = {1, 2}
BAMSE_CONF_THRESH = 0.95

# CV thresholds
MIN_AREA = 500
MAX_AREA_FRAC = 0.25
# LEGO HSV ranges
LEGO_COLORS = {
    "red":      [((5,120,70), (10,255,255)), ((170,100,100),(179,255,255))],
    "yellow":   [((15,120,120),(35,255,255))],
    "green":    [((40,80,50),(85,255,255))],
    "blue":     [((86,100,100),(140,255,255))]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Resnet18 model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Preprocess
cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# Lego detektor
def detect_lego_rois(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img_area = frame.shape[0] * frame.shape[1]
    max_area = img_area * MAX_AREA_FRAC

    detections = []

    for color_name, ranges in LEGO_COLORS.items():
        mask = None

        for lower, upper in ranges:
            part = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = part if mask is None else cv2.bitwise_or(mask, part)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if MIN_AREA < area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append({
                    "type": "lego",
                    "color": color_name,
                    "bbox": (x, y, w, h)
                })

    return detections

# Tracking af mulige bamse ROI
def propose_bamse_rois(frame):
    h, w, _ = frame.shape
    size = min(h, w) // 2
    cx, cy = w // 2, h // 2

    rois = [
        (cx-size//2, cy-size//2, size, size),
        (0, 0, size, size),
        (w-size, 0, size, size),
        (0, h-size, size, size),
        (w-size, h-size, size, size),
    ]

    return [(x,y,w,h) for x,y,w,h in rois if x>=0 and y>=0]

# CNN-klassifikation på ROIs
def classify_bamse(frame, rois):
    detections = []

    for (x, y, w, h) in rois:
        crop = frame[y:y+h, x:x+w]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue

        inp = cnn_transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)[0]

        cls = probs.argmax().item()
        conf = probs[cls].item()

        if cls in BAMSE_TARGETS and conf > BAMSE_CONF_THRESH:
            detections.append({
                "type": "bamse",
                "class": BAMSE_CLASSES[cls],
                "confidence": conf,
                "bbox": (x, y, w, h)
            })

    return detections

# Tegner detektioner
def draw_detections(frame, detections):
    for d in detections:
        x, y, w, h = d["bbox"]

        if d["type"] == "lego":
            label = f"LEGO: {d['color']} klods"
            color = (255, 0, 0)
        else:
            label = f"{d['class']} {d['confidence']:.2f}"
            color = (0, 255, 0)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, label, (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
# Main loop
cap = cv2.VideoCapture(CAMERA_INDEX)
assert cap.isOpened(), "Kan ikke åbne webcam"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    scale = TARGET_WIDTH / frame.shape[1]
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    detections = []
    detections += detect_lego_rois(frame)
    detections += classify_bamse(frame, propose_bamse_rois(frame))

    draw_detections(frame, detections)

    cv2.imshow("ToyIDHelper", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
  

