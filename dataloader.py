from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import torch

def get_transform():
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_dataset(cfg, transform):
    train_path = r"C:\Users\User\PycharmProjects\PythonProject\data\cropped_training_filtered"
    test_path = r"C:\Users\User\PycharmProjects\PythonProject\data\Validation_filtered"

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    train_dataset, valid_dataset = random_split(train_dataset, (int(len(train_dataset) * cfg.data.train_ratio), len(train_dataset) - int(len(train_dataset) * cfg.data.train_ratio)))
    test_dataset = datasets.ImageFolder(test_path, transform=transform)

    return train_dataset, valid_dataset, test_dataset

def get_dataloader(cfg, train_dataset, valid_dataset, test_dataset):

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    return train_loader, valid_loader, test_loader

