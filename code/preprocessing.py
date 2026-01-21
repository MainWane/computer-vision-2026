# Billede preprocess
from torchvision import transforms

def get_transforms(mode='train', image_size=(64, 64)):
    """
    Returnerer en torchvision.transforms.Compose baseret på mode.

    Args:
        mode (str): 'train' eller 'test'
        image_size (tuple): størrelse som billederne skal skaleres til

    Returns:
        torchvision.transforms.Compose: transformations pipeline
    """
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )

        ])
    elif mode == 'test':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'test'.")
    
    return transform
