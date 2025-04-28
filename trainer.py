import torch
import logging


def train(model, train_loader, valid_loader, criterion, optimizer, epochs, writer, device):
    best_accuracy = 0.0
    best_model_weights = None
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('Accuracy/train', accuracy, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_weights = model.state_dict().copy()

        logging.info(f"Epoch [{epoch+1}/{epochs}], Train_Loss: {avg_train_loss:.4f}, Val_Accuracy: {accuracy:.4f}")

        model.train()

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        torch.save(model.state_dict(), 'best_model.pth')
        logging.info(f"Best model weights saved with accuracy: {best_accuracy:.2f}")

def evaluate(model, test_loader, writer, device):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    writer.add_scalar('Accuracy/test', accuracy, len(test_loader))
    logging.info(f"Test Accuracy: {accuracy:.2f}")

    writer.close()