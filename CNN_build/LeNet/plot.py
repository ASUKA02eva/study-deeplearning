from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as data
import numpy as np

train_dataset = FashionMNIST(root='./data',
                             train=True,
                             transform=transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]),
                             download = True
                             
                             )

train_dataset = data.DataLoader(train_dataset, batch_size=64, shuffle=True)