import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Model ------------------
model_path = Path("bamsenet_resnet18.pt")
assert model_path.exists(), "Model mangler"

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

classes = ["NotBamse", "panda", "ugli"]
TARGET_CLASSES = {1, 2}
CONF_THRESH = 0.10

# ------------------ Preprocess ------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

def get_crops_with_positions(frame, size=64):
    h, w, _ = frame.shape
    crops = []
    positions = []

    cx, cy = w // 2, h // 2

    # center
    crops.append(frame[cy-size//2:cy+size//2, cx-size//2:cx+size//2])
    positions.append((cx-size//2, cy-size//2))

    # corners
    crops.append(frame[0:size, 0:size])
    positions.append((0, 0))

    crops.append(frame[0:size, w-size:w])
    positions.append((w-size, 0))

    crops.append(frame[h-size:h, 0:size])
    positions.append((0, h-size))

    crops.append(frame[h-size:h, w-size:w])
    positions.append((w-size, h-size))

    valid = [
        (c, p) for c, p in zip(crops, positions)
        if c.shape[0] == size and c.shape[1] == size
    ]

    crops, positions = zip(*valid)
    return crops, positions

# ------------------ Webcam ------------------
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Kan ikke åbne webcam"

with torch.inference_mode():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        crops, positions = get_crops_with_positions(frame)
        batch = torch.stack([transform(c) for c in crops]).to(device)

        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

        crop_conf, crop_cls = probs.max(dim=1)
        best_crop = crop_conf.argmax().item()

        conf = crop_conf[best_crop].item()
        cls  = crop_cls[best_crop].item()

        if cls in TARGET_CLASSES and conf > CONF_THRESH:
            x, y = positions[best_crop]
            label = f"{classes[cls]} {conf:.2f}"

            cv2.rectangle(
                frame,
                (x, y),
                (x + 64, y + 64),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                label,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("BamseSpotter (single-pass)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
