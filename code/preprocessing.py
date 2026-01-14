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
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
    elif mode == 'test':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'test'.")
    
    return transform
