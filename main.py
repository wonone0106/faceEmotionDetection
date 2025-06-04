import hydra
from omegaconf import OmegaConf
import torch.optim as optim
from dataloader import get_transform, get_dataloader, get_dataset
from model import EmotionMobileNet
import logging
import torch.nn as nn
from trainer import train, evaluate
import torch
import os

@hydra.main(version_base=None, config_path="./config", config_name="train.yaml")
def main(cfg):
    OmegaConf.to_yaml(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    transform = get_transform()

    train_dataset, valid_dataset, test_dataset = get_dataset(cfg, transform)
    logging.info(f"train_dataset: {len(train_dataset)}, valid_dataset: {len(valid_dataset)}, test_dataset: {len(test_dataset)}")

    train_loader, valid_loader, test_loader = get_dataloader(cfg, train_dataset, valid_dataset, test_dataset)
    logging.info(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}, test_loader: {len(test_loader)}")

    model = EmotionMobileNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    epoch = 0

    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth")
        logging.info(f"Loading checkpoint successfully")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logging.info(f"Resuming training from epoch {epoch+1} with loss {loss:.4f}")

    train(model, train_loader, valid_loader, criterion, optimizer, epoch, cfg.train.epochs, device)
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()


