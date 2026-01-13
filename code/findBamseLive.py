import cv2
import torch
from torchvision import transforms
from PIL import Image
from main import Net  # din Net-klasse
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Net().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.eval()

# Klassenavne
classes = ['ugli', 'ugli2', 'ugli3']

# Transformationer
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Kan ikke åbne kamera")

TARGET_WIDTH = 600
min_area = 500

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    scale = TARGET_WIDTH / frame.shape[1]
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # Konverter til RGB for PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Simpel "sliding window" approach: crop hele frame i et stykke ---
    pil_img = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)  # batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1, keepdim=True).item()

    # Hvis modelens confidence er høj nok (log_softmax), vis bounding box hele frame
    prob = torch.exp(output)[0, pred].item()
    if prob > 0.3:  # threshold, juster som du vil
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
        cv2.putText(frame, f"{classes[pred]} ({prob:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC for exit
        break

cap.release()
cv2.destroyAllWindows()
