import torch
from torchvision import transforms
from PIL import Image
from main import Net  # hvis din Net-klasse er i main.py
from pathlib import Path

# Vælg device
device = torch.device("cpu")

# Load modellen
model = Net().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.eval()  # vigtigt: sætter dropout/batchnorm i eval-mode

# Load og preprocess billedet
image_path = "C:/Users/ulrik/Desktop/cvmaterial/u1.jpg" 
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # samme som under træning
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # samme som træning
])

image = Image.open(image_path).convert("RGB")  # sikrer 3 kanaler
image = transform(image).unsqueeze(0)  # tilføjer batch-dimension
image = image.to(device)

# Kør forward pass
with torch.no_grad():
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)

classes = ['ugli', 'ugli2', 'ugli3']
print("Predicted class:", classes[pred.item()])