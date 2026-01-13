import  os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from pathlib import Path
'''
Dette er den lidt lettere måde at lave datasæt på end den indbyggede metode i PyTorch.
Det er lettere fordi vi bare ændre i stierne hvis vi vil have andet data.
'''

# Hvilke mapper vores billededata ligger i
BASE_DIR = Path(__file__).resolve().parents[2]
trainDir = BASE_DIR / "code" / "data" / "train"
testDir  = BASE_DIR / "code" / "data" / "test"


# # Hvordan vi vil behandle billederne. Her bare en resize til at gøre alle billederne mindre, og random flips.
# transform = transforms.Compose([
#             transforms.Resize(size=(64,64)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.3),
#             transforms.ToTensor()
    
#             ])

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

trainData = datasets.ImageFolder(root=trainDir,
                                 transform = train_transform,
                                 target_transform=None)
print(f"Train data:\n{trainData}")

testData = datasets.ImageFolder(root=testDir,
                                transform = test_transform,
                                target_transform=None)

print(f"Test data:\n{testData}")
print(trainData.classes)
print(trainData.class_to_idx)
print(testData.class_to_idx)
