from scipy.cluster.hierarchy import weighted
from torch import nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.baseResNet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.baseResNet.fc = nn.Sequential(
            nn.Linear(self.baseResNet.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.baseResNet(x)
        return x


class EmotionMobileNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionMobileNet, self).__init__()
        self.base_model = models.mobilenet_v2(weights=None)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.base_model.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)