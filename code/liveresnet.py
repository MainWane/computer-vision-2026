import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# ---------- Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("bamsenet_resnet182.pt", map_location=device))
model.to(device)
model.eval()

classes = ['NotBamse', 'panda', 'ugli']
TARGET_CLASSES = {1, 2}  # panda, ugli
CONF_THRESH = 0.97

# ---------- Preprocess ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)

WINDOW = 96
STRIDE = 48

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    for y in range(0, h - WINDOW, STRIDE):
        for x in range(0, w - WINDOW, STRIDE):
            crop = frame[y:y+WINDOW, x:x+WINDOW]
            inp = transform(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(inp)
                probs = torch.softmax(logits, dim=1)[0]
                cls = probs.argmax().item()
                conf = probs[cls].item()

            if cls in TARGET_CLASSES and conf > CONF_THRESH:
                label = f"{classes[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x, y), (x+WINDOW, y+WINDOW), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("BamseSpotter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

print(model.fc.weight.abs().mean())

