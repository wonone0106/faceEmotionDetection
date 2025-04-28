import kagglehub
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, random_split

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
            transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    return transform

def get_dataset(cfg, transform):
    path = r"C:\Users\User\PycharmProjects\PythonProject\data\Validation"

    train_dataset = datasets.ImageFolder(path, transform=transform)
    train_dataset, valid_dataset = random_split(train_dataset, (int(len(train_dataset) * cfg.data.train_ratio), len(train_dataset) - int(len(train_dataset) * cfg.data.train_ratio)))
    valid_dataset, test_dataset = random_split(valid_dataset, (int(len(valid_dataset) * cfg.data.valid_ratio),len(valid_dataset) - int(len(valid_dataset) * cfg.data.valid_ratio)))

    return train_dataset, valid_dataset, test_dataset

def get_dataloader(cfg, train_dataset, valid_dataset, test_dataset):

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

