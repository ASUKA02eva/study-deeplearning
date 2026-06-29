import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from ResNetModel import ResNet18, Residual
import time
import pandas as pd
import copy
import os


def train_val_data_process():
    train_dataset = FashionMNIST(root='./data_Fashion-mnist',
                             train=True,
                             transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
                             download = True
                             )
    
    train_data,val_data = data.random_split(train_dataset,[round(len(train_dataset)*0.8),round(len(train_dataset)*0.2)])
    
    train_dataloader = data.DataLoader(train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=0)
    
    val_dataloader = data.DataLoader(val_data,
                                     batch_size=64,
                                     shuffle=True,
                                     num_workers=0)
    
    
    return train_dataloader,val_dataloader    


def train_model_process(model,train_dataloader,val_dataloader,epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model = model.to(device)
    
    best_weights = copy.deepcopy(model.state_dict())
    
    best_acc = 0.0
    
    train_loss_all = []
    val_loss_all = []
    
    train_acc_all = []
    val_acc_all = []
    
    
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-"*10)
        since = time.time()
        #初始化参数
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        
        #训练集和验证集的数量
        train_num = 0
        val_num = 0
        
        for step,(inputs,labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            model.train()
            
            outputs = model(inputs)
            
            pre_lab = torch.argmax(outputs,dim=1)
            
            loss = criterion(outputs,labels)
            
            #一个batch结束，梯度清零
            optimizer.zero_grad()
            
            #反向传播
            loss.backward()
            
            optimizer.step()
            #累加损失
            train_loss += loss.item() * inputs.size(0)
            #累加正确预测的数量
            train_corrects += torch.sum(pre_lab == labels.data)
            #累加训练集的数量
            train_num += inputs.size(0)   
         
        with torch.no_grad():    
            for step,(inputs,labels) in enumerate(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                model.eval()
                
                outputs = model(inputs)           
                pre_lab = torch.argmax(outputs,dim=1)        
                loss = criterion(outputs,labels)        
                #一个batch结束，梯度清零
                
                #累加损失
                val_loss += loss.item() * inputs.size(0)
                #累加正确预测的数量
                val_corrects += torch.sum(pre_lab == labels.data)
                #累加验证集的数量
                val_num += inputs.size(0)
            
            
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append((train_corrects.double()/train_num).item())
        val_acc_all.append((val_corrects.double()/val_num).item())    
        
        print(f"train_loss:{train_loss/train_num:.4f} train_acc:{train_corrects.double()/train_num:.4f} val_loss:{val_loss/val_num:.4f} val_acc:{val_corrects.double()/val_num:.4f}")
            
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_weights = copy.deepcopy(model.state_dict())

        time_duration = time.time() - since
        print(f"Time duration: {time_duration//60:.0f}m {time_duration%60:.0f}s")

    torch.save(best_weights, r"D:\Codes\VSCode\DeepLearning\CNN_build\ResNet\ResNet.pth")
    
    train_process = pd.DataFrame({"epoch":range(epochs),
                                  "train_loss":train_loss_all,
                                  "val_loss":val_loss_all,
                                  "train_acc":train_acc_all,
                                  "val_acc":val_acc_all})
    return train_process



def plot_train_process(train_process):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process["train_loss"],label="train_loss")
    plt.plot(train_process["epoch"],train_process["val_loss"],label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"],train_process["train_acc"],label="train_acc")
    plt.plot(train_process["epoch"],train_process["val_acc"],label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(r"C:\Users\徐林智\Desktop\ResNet.png")
    plt.close()
    


if __name__ == "__main__":
    model = ResNet18()
    train_dataloader,val_dataloader = train_val_data_process()
    train_process = train_model_process(model,train_dataloader,val_dataloader,epochs=20)    
    plot_train_process(train_process)
