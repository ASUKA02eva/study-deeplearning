import torch
import torch.nn as nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,stride=1,padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5,stride=1,padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        
        self.flatten = nn.Flatten()
        
        self.fulllayer1 = nn.Linear(400, 120)
        self.fulllayer2 = nn.Linear(120, 84)
        self.fulllayer3 = nn.Linear(84, 10)
        
        self.sigmoid=nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool2(x)
        x = self.sigmoid(x)
        
        x = self.flatten(x)
        
        x = self.fulllayer1(x)
        x = self.sigmoid(x)
        x = self.fulllayer2(x)
        x = self.sigmoid(x)
        x = self.fulllayer3(x)

        return x
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LeNet().to(device)
    
    print(summary(model, input_size=(1, 28, 28)))