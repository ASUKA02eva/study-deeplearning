import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fulllayer1 = nn.Linear(256*6*6, 4096)
        self.fulllayer2 = nn.Linear(4096, 4096)
        self.fulllayer3 = nn.Linear(4096, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        
        x = self.fulllayer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fulllayer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fulllayer3(x)

        return x
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AlexNet().to(device)
    
    print(summary(model, input_size=(1, 224, 224)))