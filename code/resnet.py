import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from dataloader import trainData, testData
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze alle feature layers
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True


# Erstat fc-laget så det passer til dine 3 klasser
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model = model.to(device)

optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])

writer = SummaryWriter(log_dir="runs/bamsenet")

train_loader = DataLoader(trainData, batch_size=64, shuffle=True)
test_loader = DataLoader(testData, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()

# Simpel træningsloop
for epoch in range(14):
    model.train()
    running_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)

    print(f"Epoch {epoch+1}/{epoch} – Train loss: {avg_loss:.4f}")



    # Eval
model.eval()
    
correct = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = output.argmax(dim=1)

        correct += preds.eq(target).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    accuracy = correct / len(test_loader.dataset)
    writer.add_scalar("Accuracy/test", accuracy, epoch)

    print(f"Final TEST accuracy: {accuracy:.2%}")

# Confusion matrix
cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=trainData.classes
)
fig, ax = plt.subplots(figsize=(4,4))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.tight_layout()

writer.add_figure("ConfusionMatrix/test", fig, epoch)
writer.close()
plt.close(fig)
torch.save(model.state_dict(), "bamsenet_resnet182.pt")