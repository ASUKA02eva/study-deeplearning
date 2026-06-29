# DeepLearning 项目知识点总览

> 本仓库记录了作者从零入门深度学习的两条学习路径：
>
> 1. **`CNN_build/`** —— 用 PyTorch 从零搭建 5 个经典卷积神经网络（LeNet / AlexNet / VGGNet / GoogLeNet / ResNet），统一在 Fashion-MNIST 上训练与评测，直观感受 CNN 架构十年演进。
> 2. **`cs231n/`** —— 完成 Stanford CS231n Assignment 1，从 KNN 一路手写到 BatchNorm/Dropout/Adam，亲手实现反向传播。详细笔记见 [`cs231n/learning_notes.md`](cs231n/learning_notes.md)。
>
> 本文档聚焦 `CNN_build/` 的架构知识点与两条路径中通用的工程实践，并补全 cs231n 笔记未展开的卷积网络理论。

---

## 目录

- [一、项目结构](#一项目结构)
- [二、通用训练流程（五个网络共用）](#二通用训练流程五个网络共用)
- [三、卷积神经网络的核心组件](#三卷积神经网络的核心组件)
- [四、五个经典架构详解](#四五个经典架构详解)
  - [4.1 LeNet-5（1998）](#41-lenet-51998)
  - [4.2 AlexNet（2012）](#42-alexnet2012)
  - [4.3 VGGNet（2014）](#43-vggnet2014)
  - [4.4 GoogLeNet / Inception v1（2014）](#44-googlenet--inception-v12014)
  - [4.5 ResNet-18（2015）](#45-resnet-182015)
- [五、架构对比总表](#五架构对比总表)
- [六、关键知识点索引](#六关键知识点索引)
- [七、学习路线与延伸](#七学习路线与延伸)

---

## 一、项目结构

```
DeepLearning/
├── CNN_build/                       # PyTorch 经典 CNN 实现
│   ├── LeNet/        {Model.py, train.py, test.py}
│   ├── AlexNet/      {Model.py, train.py, test.py}
│   ├── VGGNet/       {Model.py, train.py, test.py}
│   ├── GoogLeNet/    {Model.py, train.py, test.py}
│   └── ResNet/       {Model.py, train.py, test.py}
├── cs231n/
│   ├── learning_notes.md            # Assignment 1 详细笔记
│   └── assignments/assignment1/     # KNN→神经网络源码与 ipynb
└── .gitignore                       # 忽略数据集与 .pth 权重
```

每个网络目录下的文件职责一致：

| 文件 | 作用 |
|---|---|
| `*Model.py` | 定义网络结构（`nn.Module` 子类），可单独运行查看 `torchsummary` |
| `train.py` | 数据加载、训练循环、保存验证集最优权重、绘制 loss/acc 曲线 |
| `test.py` | 加载权重，在测试集上评估准确率 |

---

## 二、通用训练流程（五个网络共用）

五个网络的 `train.py` / `test.py` 几乎是同一套模板，差异只在网络结构与超参数。这套模板本身就是一份完整的"PyTorch 训练范式"教材。

### 2.1 数据：Fashion-MNIST

- 28×28 灰度图，10 个服装类别（T-shirt、Trouser、Dress、Sneaker、Bag…）
- 训练集 60000、测试集 10000
- 通过 `torchvision.datasets.FashionMNIST` 自动下载
- **Resize 策略**：LeNet 保持 28×28；AlexNet/VGG/GoogLeNet/ResNet 放大到 224×224 以匹配经典架构的输入尺寸
- **预处理**：仅 `transforms.ToTensor()`（归一化到 [0,1] 并转为 NCHW）

### 2.2 数据划分与加载

```python
train_data, val_data = data.random_split(train_dataset, [0.8, 0.2])
train_dataloader = data.DataLoader(train_data, batch_size=B, shuffle=True)
```

- 80% 训练 / 20% 验证，用 `random_split` 随机切分
- `DataLoader` 负责 batch 化、shuffle、多进程加载
- 不同网络 batch size 不同（见 [对比表](#五架构对比总表)）

### 2.3 训练循环（每个 epoch）

```
for epoch:
    for batch in train_loader:        # 训练阶段
        model.train()
        outputs = model(inputs)
        loss = CrossEntropyLoss(outputs, labels)
        optimizer.zero_grad()         # 梯度清零
        loss.backward()               # 反向传播
        optimizer.step()              # 更新参数

    with torch.no_grad():             # 验证阶段
        for batch in val_loader:
            model.eval()
            outputs = model(inputs)
            ...                       # 累计 val_loss / val_acc

    if val_acc > best_acc:            # 保存最优权重
        best_weights = deepcopy(model.state_dict())
```

### 2.4 三个关键模式开关

| 调用 | 作用 |
|---|---|
| `model.train()` | 启用 Dropout、让 BatchNorm 用 batch 统计量 |
| `model.eval()` | 关闭 Dropout、BatchNorm 用 running 统计量 |
| `with torch.no_grad():` | 停止记录计算图，节省显存、加速推理 |

**忘记切 `eval()` 是初学者最常见的 bug**——验证集准确率会异常波动，因为 Dropout 还在随机丢弃、BN 还在用当前 batch 的统计量。

### 2.5 最优权重保存与加载

- 训练时用 `copy.deepcopy(model.state_dict())` 缓存验证集最佳权重
- `torch.save(best_weights, "XxxNet.pth")` 落盘
- 测试时 `model.load_state_dict(torch.load(..., weights_only=True))`
- `.pth` 已在 `.gitignore` 中忽略（超 GitHub 100MB 限制）

### 2.6 统一超参数

| 项 | 值 |
|---|---|
| 优化器 | `torch.optim.Adam(lr=0.001)` |
| 损失函数 | `nn.CrossEntropyLoss()`（内含 LogSoftmax + NLLLoss） |
| Epoch 数 | 20 |
| 设备 | `cuda` if available else `cpu` |

> Adam + 交叉熵是分类任务最稳的默认组合，无需手调学习率即可在 Fashion-MNIST 上达到 0.9+ 测试准确率。

---

## 三、卷积神经网络的核心组件

五个架构其实就是这些组件的不同组合方式。先理解零件，再理解装配。

### 3.1 卷积层 `nn.Conv2d`

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
```

- **输出尺寸**：$H_{out} = \lfloor (H_{in} + 2P - K)/S \rfloor + 1$
- **参数量**：$K \times K \times C_{in} \times C_{out} + C_{out}$
- **感受野**：堆叠多层小卷积核等价于单层大卷积核，但参数更少、非线性更强（VGG 的核心论点）

### 3.2 池化层

| 类型 | 公式 | 特点 |
|---|---|---|
| `MaxPool2d` | 取窗口最大值 | 保留最强响应，AlexNet 起成为主流 |
| `AvgPool2d` | 取窗口均值 | 平滑，LeNet 时代常用 |
| `AdaptiveAvgPool2d((1,1))` | 全局平均池化 | 任意尺寸 → 1×1，GoogLeNet/ResNet 用来替代大全连接层 |

### 3.3 激活函数

| 激活 | 时代 | 导数 |
|---|---|---|
| `Sigmoid` | LeNet | $\sigma(1-\sigma)$，易饱和、梯度消失 |
| `ReLU` | AlexNet 起 | $\mathbb{1}(x>0)$，计算快、缓解梯度消失 |

> AlexNet 论文最重要的影响之一就是**把 Sigmoid/Tanh 换成 ReLU**——训练速度提升数倍，从此 ReLU 成为默认选择。

### 3.4 BatchNorm `nn.BatchNorm2d`

$$\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}},\quad out = \gamma\hat{x} + \beta$$

- 稳定深层网络训练、加速收敛、缓解内部协变量偏移
- **训练用 batch 统计量，测试用 running averages**——这就是为什么必须 `model.eval()`
- VGG 引入 BN 后可以用更深的网络、更大的学习率

### 3.5 Dropout `nn.Dropout(p)`

- 训练时以概率 `p` 随机置零，并按 $1/(1-p)$ 缩放保持期望
- 测试时恒等
- AlexNet 首次大规模使用，是全连接层防过拟合的标配

### 3.6 全连接层与全局平均池化

- 早期网络（LeNet/AlexNet/VGG）末尾接 `Flatten → Linear(4096) → Linear(4096) → Linear(10)`，参数量巨大
- GoogLeNet 起改用 `AdaptiveAvgPool2d((1,1)) → Linear(10)`，把几亿参数压到几十万，同时减少过拟合

---

## 四、五个经典架构详解

### 4.1 LeNet-5（1998）

> Yann LeCun 提出，第一个成功应用于手写数字识别的 CNN。

**结构**（`LeNetModel.py`）：

```
Input(1,28,28)
 → Conv2d(1→6, k=5, p=2)  → Sigmoid → AvgPool(2)      # 28→14
 → Conv2d(6→16, k=5, p=0) → Sigmoid → AvgPool(2)      # 14→5
 → Flatten → Linear(400→120) → Sigmoid
 → Linear(120→84) → Sigmoid
 → Linear(84→10)
```

**知识点**：
- **sigmoid + 平均池化**：1998 年的时代特征，现在已被 ReLU + MaxPool 取代
- **padding=2 的 5×5 卷积**：保持 28×28 尺寸，等价于 3×3 卷积的感受野扩展
- **三层全连接**：早期 CNN 末尾全连接很重，因为当时没有全局平均池化思想
- 输入保持 28×28（其余四个网络都 resize 到 224）

---

### 4.2 AlexNet（2012）

> Krizhevsky 等，ImageNet 2012 冠军，引爆深度学习浪潮。

**结构**（`AlexNetModel.py`）：

```
Input(1,224,224)
 → Conv2d(1→96, k=11, s=4, p=2) → ReLU → MaxPool(3,s=2)   # 224→27
 → Conv2d(96→256, k=5, p=2)     → ReLU → MaxPool(3,s=2)   # 27→13
 → Conv2d(256→384, k=3, p=1)    → ReLU
 → Conv2d(384→384, k=3, p=1)    → ReLU
 → Conv2d(384→256, k=3, p=1)    → ReLU → MaxPool(3,s=2)   # 13→6
 → Flatten → Linear(256*6*6→4096) → ReLU → Dropout(0.5)
          → Linear(4096→4096)     → ReLU → Dropout(0.5)
          → Linear(4096→10)
```

**知识点**：
- **ReLU 激活**：首次大规模替代 sigmoid，解决梯度饱和
- **重叠 MaxPool**（kernel=3, stride=2）：比无重叠池化（kernel=stride）提取更细
- **Dropout(0.5)**：全连接层防过拟合，首次在 CNN 中引入
- **大卷积核 11×11 + stride=4**：第一步就大幅降采样，因为输入大（224×224）
- **三层全连接 4096-4096-10**：参数量约 60M，是现代网络防过拟合的反面教材

---

### 4.3 VGGNet（2014）

> Oxford VGG 组，论证"连续小卷积核堆叠"思想。

**结构**（`VGGNetModel.py`，对应 VGG-16 风格）：

```
block1: [Conv(1→64,3,p1)→BN→ReLU]×2 → MaxPool        # 224→112
block2: [Conv(64→128,3,p1)→BN→ReLU]×2 → MaxPool      # 112→56
block3: [Conv(128→256,3,p1)→BN→ReLU]×3 → MaxPool     # 56→28
block4: [Conv(256→512,3,p1)→BN→ReLU]×3 → MaxPool     # 28→14
block5: [Conv(512→512,3,p1)→BN→ReLU]×3 → MaxPool     # 14→7
head:   Flatten → Linear(7*7*512→4096)→ReLU→Dropout
              → Linear(4096→4096)→ReLU→Dropout
              → Linear(4096→10)
```

**知识点**：
- **3×3 卷积堆叠**：两个 3×3 等价一个 5×5 的感受野，三个等价 7×7，但参数量分别从 $25C^2$、$49C^2$ 降到 $18C^2$、$27C^2$，且非线性更多
- **Block 化设计**：用 `nn.Sequential` 把"卷积+BN+ReLU"封装成块，代码可读性大幅提升
- **BatchNorm**：每个 Conv 后都加 BN，使深层网络可用更大学习率训练
- **通道数翻倍、空间尺寸减半**：经典的"金字塔"结构，信息从空间维度转移到通道维度
- **Dropout 无参数**：本实现用默认 `p=0.5`（AlexNet 风格）

---

### 4.4 GoogLeNet / Inception v1（2014）

> Google 团队，ImageNet 2014 冠军，提出 Inception 模块。

**Inception 模块**（`GoogLeNetModel.py` 中的 `Inception` 类）：

```
       输入 x
   ┌───┼───┬───┐
  1×1   1×1  1×1  MaxPool(3,p1)
  conv  ↓conv ↓conv ↓
        3×3  5×5  1×1 conv
   └───┼───┴───┴───┘
       concat (沿通道)
```

四条并行路径，输出在通道维拼接：

| 路径 | 操作 | 作用 |
|---|---|---|
| p1 | 1×1 conv | 跨通道信息融合 |
| p2 | 1×1 → 3×3 conv | 1×1 先降维，再做空间特征 |
| p3 | 1×1 → 5×5 conv | 1×1 先降维，再做更大空间特征 |
| p4 | 3×3 MaxPool → 1×1 conv | 池化后再投影 |

**整体结构**：5 个 block（b1: 7×7 stride2 + MaxPool；b2: 1×1+3×3+MaxPool；b3/b4: Inception 堆叠+MaxPool；b5: Inception + AdaptiveAvgPool + Linear(1024→10)）。

**知识点**：
- **1×1 卷积降维**：在昂贵的大卷积前先用 1×1 减少通道数，大幅降低参数与计算量。这是 Inception 系列的核心创新
- **多尺度并行**：同一层同时用 1×1/3×3/5×5/Pool 抽取不同尺度特征，让网络"自己决定"用哪种
- **全局平均池化替代大 FC**：`AdaptiveAvgPool2d((1,1))` 后直接接 10 类 Linear，参数量从几亿降到几千
- **He 初始化**（`kaiming_normal_`）：配合 ReLU 的方差缩放初始化，比 Xavier 更适合 ReLU
  ```python
  nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
  nn.init.constant_(m.bias, 0)
  ```
- 本实现省略了原论文的辅助分类器（auxiliary classifiers），主干够深即可训练

---

### 4.5 ResNet-18（2015）

> He 等，ImageNet 2015 冠军，引入残差连接，使训练上百层网络成为可能。

**残差块**（`ResNetModel.py` 中的 `Residual` 类）：

```
        x
        │
        ├─────── (identity)
        │              │
   Conv3×3 → BN → ReLU │
   Conv3×3 → BN        │
        │              │
        └─── + ────────┘ (如通道/尺寸不匹配则经 1×1 conv)
              │
            ReLU
              │
              y
```

- `use_1conv=False`：输入输出通道相同、stride=1，直接相加
- `use_1conv=True`：通道翻倍或 stride=2 下采样时，用 1×1 conv（带 stride）匹配 x 的维度

**整体结构**（ResNet-18）：

```
b1: Conv(1→64, k=7, s=2, p=3) → BN → ReLU → MaxPool(3,s=2,p1)   # 224→56
b2: Residual(64,64)  ×2                                          # 56
b3: Residual(64→128, s=2) + Residual(128,128)                    # 28
b4: Residual(128→256, s=2) + Residual(256,256)                   # 14
b5: Residual(256→512, s=2) + Residual(512,512)                   # 7
b6: AdaptiveAvgPool(1,1) → Flatten → Linear(512→10)
```

**知识点**：
- **残差连接 $y = F(x) + x$**：让梯度通过 shortcut 直接回传，解决深层网络的梯度消失/退化问题
- **1×1 卷积下采样**：当 stride≠1 或通道变化时，用 1×1 conv 配 stride 对齐 identity 维度
- **每个残差块内 BN 在 conv 后、ReLU 前**：`Conv → BN → ReLU` 是 ResNet 的标准顺序
- **stage 内通道翻倍、空间减半**：与 VGG 同构，但每个 stage 用残差块代替普通卷积
- **全局平均池化 + 单层 Linear**：和 GoogLeNet 同款轻量头
- ResNet 是第一个能稳定训练到 100+ 层的网络，从此"深"成为深度学习的常态

---

## 五、架构对比总表

| 网络 | 年份 | 输入 | 关键创新 | 激活 | 池化 | 归一化 | 防过拟合 | 末尾头 | 本实现 batch |
|---|---|---|---|---|---|---|---|---|---|
| **LeNet-5** | 1998 | 28×28 | 第一个 CNN | Sigmoid | AvgPool | — | — | 3×FC | 256 |
| **AlexNet** | 2012 | 224×224 | ReLU + 重叠MaxPool + Dropout | ReLU | MaxPool | — | Dropout(0.5) | 3×FC(4096) | 128 |
| **VGGNet** | 2014 | 224×224 | 连续 3×3 卷积堆叠 + Block 化 | ReLU | MaxPool | BatchNorm | Dropout | 3×FC(4096) | 32 |
| **GoogLeNet** | 2014 | 224×224 | Inception + 1×1 降维 + GAP | ReLU | MaxPool+GAP | — | — | GAP+FC | 64 |
| **ResNet-18** | 2015 | 224×224 | 残差连接 + 1×1 下采样 | ReLU | MaxPool+GAP | BatchNorm | — | GAP+FC | 64 |

**演进主线**：
1. 激活：Sigmoid → ReLU（解决梯度消失）
2. 归一化：无 → BatchNorm（稳定深层训练）
3. 头部：重磅全连接 → 全局平均池化 + 单层 Linear（参数骤减）
4. 连接：顺序堆叠 → 残差 shortcut（支持超深网络）
5. 卷积核：大核（11×11、7×7）→ 小核（3×3）堆叠 + 1×1 降维

---

## 六、关键知识点索引

### 6.1 PyTorch 工程实践

| 知识点 | 位置 |
|---|---|
| `nn.Module` 定义网络 / `forward` | `CNN_build/*/*Model.py` |
| `nn.Sequential` 模块化组装 | `VGGNet/GoogLeNet/ResNet/*Model.py` |
| `torchsummary.summary` 查看结构与参数量 | 所有 `*Model.py` 的 `__main__` |
| `DataLoader` + `random_split` | 所有 `train.py` |
| `model.train()` / `model.eval()` 切换模式 | 所有 `train.py` / `test.py` |
| `torch.no_grad()` 推理上下文 | 所有 `test.py` 与 `train.py` 验证段 |
| `CrossEntropyLoss`（内含 Softmax） | 所有 `train.py` |
| `Adam` 优化器 | 所有 `train.py` |
| `state_dict` 保存 / `load_state_dict` 加载 | `train.py` 保存 / `test.py` 加载 |
| `deepcopy` 保存验证集最优权重 | 所有 `train.py` |
| `to(device)` GPU/CPU 切换 | 所有文件 |
| He / Kaiming 初始化 | `GoogLeNet/GoogLeNetModel.py:74` |

### 6.2 CNN 理论

| 知识点 | 涉及网络 |
|---|---|
| 卷积参数量 / 感受野 / 输出尺寸公式 | 全部 |
| 小卷积核等价大核、参数更省 | VGG |
| 1×1 卷积降维与跨通道融合 | GoogLeNet, ResNet |
| Inception 多尺度并行 | GoogLeNet |
| 残差连接解决退化 / 梯度消失 | ResNet |
| 全局平均池化替代大 FC | GoogLeNet, ResNet |
| BatchNorm 稳定训练 | VGG, ResNet |
| Dropout 抑制全连接过拟合 | AlexNet, VGG |
| He 初始化适配 ReLU | GoogLeNet, ResNet |

### 6.3 CS231n Assignment 1 知识点

> 详见 [`cs231n/learning_notes.md`](cs231n/learning_notes.md)，这里仅做索引：

1. **KNN**：L2 距离的向量化（双循环→零循环）、多数投票
2. **线性分类器**：$f(X)=XW+b$，参数化建模的起点
3. **损失函数**：SVM Hinge Loss vs Softmax 交叉熵，数值稳定 Softmax
4. **L2 正则化**：$R(W)=\lambda\sum W^2$，抑制过拟合
5. **优化器**：SGD → Momentum → RMSProp → Adam（含偏差修正）
6. **两层网络**：affine-relu-affine-softmax，反向传播按层调用 backward
7. **深层网络 `FullyConnectedNet`**：模块化层接口
8. **BatchNorm / LayerNorm**：训练用 batch 统计、测试用 running averages
9. **Inverted Dropout**：训练除以保留概率，测试恒等
10. **梯度检验**：中心差分、相对误差 < $10^{-7}$
11. **特征工程**：HOG + 颜色直方图，对比端到端学习
12. **Solver**：模型/求解器分离，现代框架雏形

---

## 七、学习路线与延伸

### 推荐阅读顺序

1. **打基础**：先读 `cs231n/learning_notes.md`，理解反向传播、损失、优化器、归一化的从零实现
2. **学架构**：按时间顺序跑 `CNN_build/` 的五个网络，对照 [对比表](#五架构对比总表) 体会演进
3. **动手做**：
   - 用 `torchsummary` 打印每个网络，对比参数量
   - 修改 `train.py` 的 `epochs` / `batch_size` / `lr`，观察收敛曲线变化
   - 把 LeNet 的 Sigmoid 换成 ReLU、AvgPool 换成 MaxPool，看准确率提升

### 每个网络对应的原始论文

| 网络 | 论文 | 年份 |
|---|---|---|
| LeNet-5 | *Gradient-Based Learning Applied to Document Recognition* (LeCun et al.) | 1998 |
| AlexNet | *ImageNet Classification with Deep CNNs* (Krizhevsky et al.) | 2012 |
| VGGNet | *Very Deep Convolutional Networks for Large-Scale Image Recognition* (Simonyan & Zisserman) | 2014 |
| GoogLeNet | *Going Deeper with Convolutions* (Szegedy et al.) | 2014 |
| ResNet | *Deep Residual Learning for Image Recognition* (He et al.) | 2015 |

### 下一步可扩展方向

- **数据增强**：在 `transforms.Compose` 中加 RandomCrop / RandomFlip / Normalize
- **学习率调度**：`torch.optim.lr_scheduler.StepLR` / `CosineAnnealingLR`
- **迁移学习**：`torchvision.models` 加载预训练权重，迁移到 Fashion-MNIST
- **TensorBoard / W&B**：替代 `matplotlib` 保存曲线，实现训练可视化
- **CS231n Assignment 2**：从零实现卷积层、BN 层、PyTorch 风格的 autograd

---

> **一句话总结**：`cs231n/` 教你"为什么这么算"——亲手写反向传播；`CNN_build/` 教你"怎么搭起来"——把零件拼成五个里程碑架构。两条路径合起来，就是一份从底层数学到工程实践的完整入门地图。
