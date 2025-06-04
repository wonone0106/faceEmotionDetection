from scipy.cluster.hierarchy import weighted
from torch import nn
from torchvision import models

class EmotionMobileNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionMobileNet, self).__init__()
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.base_model.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)