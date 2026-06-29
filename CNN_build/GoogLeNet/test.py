import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from GoogLeNetModel import GoogLeNet
import time
import pandas as pd
import copy
import os


def test_data_process():
    test_dataset = FashionMNIST(root='./data',
                            train=False,
                            transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
                            download=True
                            )
    
    test_dataloader = data.DataLoader(test_dataset,
                                      batch_size=64,
                                      shuffle=False,
                                      num_workers=0)
    
    return test_dataloader


def test_model_process(model, test_data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = torch.nn.CrossEntropyLoss()
    
    model = model.to(device)
    
    test_corrects = 0
    
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            pre_lab = torch.argmax(outputs, dim=1)
            
            test_corrects += torch.sum(pre_lab == labels.data)
    
    test_num = len(test_data_loader.dataset)
    
    print(f"test_acc:{test_corrects.double()/test_num:.4f}")
    
    
if __name__ == "__main__":
    model = GoogLeNet()
    model.load_state_dict(torch.load(r"D:\Codes\VSCode\DeepLearning\CNN_build\GoogLeNet\GoogLeNet.pth", weights_only=True))
    test_data_loader = test_data_process()
    test_model_process(model, test_data_loader)