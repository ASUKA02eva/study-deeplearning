# CS231n 学习历程：从 KNN 到神经网络

> 这是一份关于 Stanford CS231n（深度学习与计算机视觉）Assignment 1 的学习笔记。在这份作业里，我从一个只会算距离的 KNN 开始，一步步走到能训练多层神经网络、做批归一化、用 Adam 优化的地步。写下这篇博客，既是复盘，也希望能帮到同样在入门路上的人。

---

## 写在前面

CS231n 的 Assignment 1 表面上是"实现几个分类器"，但它真正教的是**如何从零搭建一套可训练的神经网络框架**。整套作业的代码组织成一个 `cs231n` 包：

```
cs231n/
├── classifiers/       # 各种分类器
│   ├── k_nearest_neighbor.py
│   ├── linear_classifier.py
│   ├── softmax.py
│   └── fc_net.py
├── layers.py          # 原子层（affine、relu、batchnorm、dropout...）
├── layer_utils.py     # 组合层（affine-relu）
├── optim.py           # 优化器（SGD、Momentum、RMSProp、Adam）
├── solver.py          # 训练循环
├── data_utils.py      # 数据加载与预处理
├── gradient_check.py  # 梯度检验
└── features.py        # 传统图像特征（HOG、颜色直方图）
```

这条学习路线是层层递进的：**KNN → 线性分类器 → SVM/Softmax 损失 → 两层神经网络 → 深层网络 → 优化算法 → 归一化与 Dropout**。下面我按这个顺序讲。

---

## 一、KNN：最"懒"的分类器

### 思想

KNN（K-Nearest Neighbors）是入门的第一个分类器，它"训练"阶段什么都不做——只是把训练数据记下来。真正的计算都发生在预测时：对新样本算它到所有训练样本的距离，取最近的 k 个，投票决定类别。

这让我第一次直观感受到：**有些算法根本没有"学习"过程，只是靠记忆和距离**。

### L2 距离

最核心的就是距离计算。对测试点 $X[i]$ 和训练点 $X_{train}[j]$，L2（欧氏）距离是：

$$d(i, j) = \sqrt{\sum_{d=1}^{D} (X[i,d] - X_{train}[j,d])^2}$$

最朴素的实现是双层循环（`k_nearest_neighbor.py` 的 `compute_distances_two_loops`），但这样太慢。作业要求我们写出三个版本：双循环、单循环、零循环。**零循环版本**是最妙的——利用展开式：

$$\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2a^T b$$

于是整个距离矩阵可以一次矩阵乘法搞定：

$$d_{ij} = \sqrt{\sum_d X[i,d]^2 + \sum_d X_{train}[j,d]^2 - 2\, X[i] \cdot X_{train}[j]}$$

这一刻我真正理解了什么叫"向量化"——把 Python 循环换成 numpy 的矩阵运算，速度能差几百倍。

### 投票

找到 k 个最近邻后，用 `np.argsort` 排序取前 k 个标签，再用 `np.bincount` + `argmax` 多数投票：

```python
k_nearest = np.argsort(dists[i, :])[:k]
closest_y = self.y_train[k_nearest].tolist()
y_pred[i] = np.argmax(np.bincount(closest_y))
```

### 为什么 KNN 不行

KNN 在 CIFAR-10 上准确率只有 20%~28% 左右。原因很深刻：**像素距离根本不对应语义距离**。一张图平移几个像素，对人来说还是同一张图，但 L2 距离可能完全变了。这迫使我们去寻找真正"学习"的方法。

---

## 二、线性分类器：参数化建模的起点

### 从记忆到学习

KNN 的问题在于它没有"模型"。线性分类器引入了**参数**：用一个权重矩阵 $W$ 把图像直接映射成各类别的分数。

$$f(X) = XW + b$$

其中 $X \in \mathbb{R}^{N\times D}$（N 个样本，每个 D 维），$W \in \mathbb{R}^{D\times C}$（C 个类），$b \in \mathbb{R}^C$。

预测就是取分数最大的类：$y_i = \arg\max_c\, s_{i,c}$。

这个思路的转变至关重要：**我们不再记忆数据，而是学习一组参数**。一旦学好了，训练数据就可以丢掉——模型已经"浓缩"进了 $W$ 里。

### 训练：SGD

怎么学这组参数？用**随机梯度下降**（`linear_classifier.py` 的 `train` 方法）：

1. 从训练集随机抽一个 mini-batch
2. 算这个 batch 上的 loss 和梯度
3. 更新 $W \leftarrow W - \eta\, \nabla W$
4. 重复

```python
random_img = np.random.choice(num_train, batch_size, replace=True)
X_batch, y_batch = X[random_img], y[random_img]
loss, grad = self.loss(X_batch, y_batch, reg)
self.W -= learning_rate * grad
```

这里的 `loss` 方法由子类实现——可以是 SVM，也可以是 Softmax。这就是**面向对象的多态**在 ML 里的典型用法。

---

## 三、损失函数：告诉模型"错得有多离谱"

### 多类 SVM 损失（Hinge Loss）

SVM 的想法很直接：**正确类的分数必须比其他类高出至少一个安全间隔 $\Delta$（通常取 1），否则就要惩罚**。

$$L_i = \sum_{j \neq y_i} \max(0,\ s_{i,j} - s_{i,y_i} + \Delta)$$

$$L = \frac{1}{N}\sum_i L_i + \lambda R(W)$$

这个 $\max(0, \cdot)$ 叫 hinge 函数，图像是一个折线。它只在"间隔被违反"时才产生损失，否则为零——这就是所谓的"max-margin"思想。

### Softmax 损失（交叉熵）

Softmax 更"软"：它把分数通过指数归一化变成概率，然后最小化负对数似然。

**数值稳定的 Softmax**（先减最大值防溢出）：

$$p_{i,j} = \frac{e^{s_{i,j} - \max_j s_{i,j}}}{\sum_k e^{s_{i,k} - \max_j s_{i,j}}}$$

**损失**：

$$L = -\frac{1}{N}\sum_i \log p_{i,y_i} + \lambda \sum_{k,l} W_{k,l}^2$$

第一项就是**交叉熵**。当我第一次在代码里写出 `loss -= logp[y[i]]` 时，突然理解了：所谓"训练分类器"，本质上就是在让正确类的概率逼近 1。

### 梯度的优雅形式

Softmax 损失对分数的梯度非常漂亮：

$$\frac{\partial L}{\partial s_{i,j}} = \frac{1}{N}\begin{cases} p_{i,j} - 1 & j = y_i \\ p_{i,j} & j \neq y_i \end{cases}$$

向量化实现只需要两行：

```python
dZ = p.copy()
dZ[np.arange(num_train), y] -= 1
dW = X.T.dot(dZ) / num_train + 2 * reg * W
```

这就是"预测概率减去真实标签（one-hot）"——一个贯穿深度学习的经典结论。

### SVM vs Softmax

| | SVM | Softmax |
|---|---|---|
| 损失含义 | 间隔被违反才惩罚 | 概率偏离真值就惩罚 |
| 对分数敏感度 | 只关心"够不够远" | 全程关心概率分布 |
| 输出可解释性 | 分数，无概率含义 | 直接是概率 |

实际中 Softmax 用得更多，因为它输出的概率更直观。

---

## 四、L2 正则化：别太自信

如果不加约束，模型可以把训练集的分数推到无限大来"压低"损失，导致过拟合。**L2 正则化**通过惩罚大权重来抑制这种倾向：

$$R(W) = \lambda \sum_{k,l} W_{k,l}^2, \quad \nabla_W R = 2\lambda W$$

总损失：$L_{total} = L_{data} + R(W)$。

代码里有个小坑：`fc_net.py` 里用的是 `0.5 * reg * sum(W*W)`，系数 0.5 是为了让梯度正好是 $\lambda W$（消去链式求导带出的 2）。而 `softmax.py` 里直接用 `reg * sum(W*W)`，梯度就是 $2\lambda W$。两种写法都对，但要和梯度检验的约定保持一致。

---

## 五、优化与梯度下降的工程化

### 仿射层（Affine / 全连接）

神经网络的基本砖块。前向就是把输入展平再线性变换：

$$out = x_{reshape}\, W + b$$

反向由链式法则给出：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial out} W^T, \quad \frac{\partial L}{\partial W} = x^T \frac{\partial L}{\partial out}, \quad \frac{\partial L}{\partial b} = \sum_N \frac{\partial L}{\partial out}$$

### ReLU 激活

光有线性层堆叠还是线性模型，必须加非线性。ReLU 是最简单也最好用的：

$$out = \max(0, x), \quad \frac{\partial L}{\partial x} = \frac{\partial L}{\partial out} \cdot \mathbb{1}(x > 0)$$

反向传播就是"负数位置梯度清零"，一行代码：`dx = dout * (x > 0)`。

### 组合层

`layer_utils.py` 把 `affine → relu` 封成一个组合层，前向调 `affine_relu_forward`，反向调 `affine_relu_backward`。这种**模块化设计**让我体会到：好的代码结构能让复杂的网络拼装像搭积木一样。

---

## 六、两层神经网络：第一次拼出完整网络

`TwoLayerNet` 是第一个"真正"的神经网络。架构是：

$$\text{affine} - \text{relu} - \text{affine} - \text{softmax}$$

前向：

$$h = \text{ReLU}(XW_1 + b_1), \quad s = hW_2 + b_2$$

损失：

$$L = L_{softmax}(s, y) + \frac{1}{2}\lambda(\|W_1\|_F^2 + \|W_2\|_F^2)$$

反向是这个作业最爽的部分——**只要前面的层都写对了，反向就是按顺序调用各层的 backward**：

```python
data_loss, dscores = softmax_loss(scores, y)
dout1, dW2, db2 = affine_backward(dscores, cache2)
dX, dW1, db1 = affine_relu_backward(dout1, cache1)
dW2 += reg * W2
dW1 += reg * W1
```

这就是**反向传播的精髓**：梯度从输出端一路传回输入端，每层只需关心自己怎么把上游梯度传给下游。当我跑通梯度检验、看到相对误差小于 $10^{-7}$ 时，那一刻的成就感难以言表。

---

## 七、深层网络与模块化架构

`FullyConnectedNet` 把两层网络推广到任意层，架构是：

$$\{\text{affine} - [\text{batch/layer norm}] - \text{relu} - [\text{dropout}]\}\times(L-1) - \text{affine} - \text{softmax}$$

每个 `{...}` 块重复 $L-1$ 次。这让我意识到：**深度学习框架的本质就是一套可组合的层接口**。PyTorch、TensorFlow 无非是把这套抽象做得更通用而已。

权重初始化用高斯分布 $W \sim \mathcal{N}(0, weight\_scale^2)$，偏置初始化为零。`weight_scale` 不能太大也不能太小——大了梯度爆炸，小了信号消失。这是第一次直面"初始化"这个坑。

---

## 八、归一化：让训练更稳

### 批归一化（Batch Normalization）

深层网络有个老大难问题：内部协变量偏移（Internal Covariate Shift）——每层输入的分布在训练中不断变化。**Batch Norm** 的解法是在每个 mini-batch 上把每维特征归一化到零均值、单位方差，再用可学习的 $\gamma, \beta$ 缩放平移：

$$\mu = \frac{1}{N}\sum_i x_i, \quad \sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad out = \gamma \hat{x}_i + \beta$$

**训练用 batch 统计量，测试用 running averages**：

$$\mu_{run} \leftarrow m\cdot\mu_{run} + (1-m)\cdot\mu$$

这解释了为什么 BN 层要分 `train` / `test` 模式——测试时一个样本没法算 batch 统计量。

反向传播的简化形式非常优雅：

$$\frac{\partial L}{\partial x_i} = \frac{1}{N\sqrt{\sigma^2+\epsilon}}\left(N\frac{\partial L}{\partial \hat{x}_i} - \sum_j \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i\sum_j \frac{\partial L}{\partial \hat{x}_j}\hat{x}_j\right)$$

### 层归一化（Layer Normalization）

BN 的痛点是对 batch size 敏感。**Layer Norm** 改成对每个样本（行方向）归一化：

$$\mu_i = \frac{1}{D}\sum_d x_{i,d},\quad \hat{x}_{i,d} = \frac{x_{i,d}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}$$

训练和测试行为一致，无需 running averages。LN 在 RNN/Transformer 里更常用——这让我明白：**没有银弹，不同场景选不同归一化**。

---

## 九、Dropout：随机失活防过拟合

**Inverted Dropout** 是正则化的另一招。训练时随机"丢弃"一部分神经元，测试时不动：

**训练**：

$$mask = \frac{\mathbb{1}(\text{rand}() < p)}{p},\quad out = x \cdot mask$$

**测试**：$out = x$

关键是那个 `/p`——**训练时除以保留概率 p，使得期望保持不变**，测试时就不用额外缩放。这种"inverted"设计比原始 dropout 更优雅，是现代实现的标准做法。

反向传播：$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial out}\cdot mask$，梯度只在被保留的神经元上流动。

---

## 十、优化器：比 SGD 更聪明的更新

`optim.py` 里实现了四种优化器，这是一个从"蛮力"到"自适应"的演进：

### 1. Vanilla SGD

$$w \leftarrow w - \eta\, dw$$

最朴素，但容易在峡谷形 loss 里震荡。

### 2. SGD + Momentum

$$v \leftarrow \mu v - \eta\, dw, \quad w \leftarrow w + v$$

引入"动量"——把历史梯度方向累加起来，像球滚下坡一样有惯性，能冲出局部震荡。

### 3. RMSProp

$$cache \leftarrow \rho\, cache + (1-\rho)\, dw^2$$

$$w \leftarrow w - \eta\,\frac{dw}{\sqrt{cache} + \epsilon}$$

**自适应学习率**：梯度大的参数步长自动变小，梯度小的自动变大。用指数加权移动平均缓存二阶矩。

### 4. Adam

$$m \leftarrow \beta_1 m + (1-\beta_1)dw,\quad v \leftarrow \beta_2 v + (1-\beta_2)dw^2$$

$$\hat{m} = \frac{m}{1-\beta_1^t},\quad \hat{v} = \frac{v}{1-\beta_2^t}$$

$$w \leftarrow w - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}$$

Adam = Momentum + RMSProp + **偏差修正**（$\hat{m}, \hat{v}$ 那两项）。偏差修正在训练初期至关重要——否则初始的零向量会让估计严重偏低。这是目前最常用的优化器。

---

## 十一、梯度检验：信任但要验证

写完反向传播，怎么知道自己没写错？**梯度检验**用数值方法独立估算梯度，和解析梯度对比：

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + h) - f(x_i - h)}{2h},\quad h \approx 10^{-5}$$

用**中心差分**而非前向差分，精度更高（$O(h^2)$ vs $O(h)$）。衡量误差用相对误差：

$$\text{rel\_error} = \frac{|g_{num} - g_{ana}|}{|g_{num}| + |g_{ana}|}$$

经验法则：相对误差小于 $10^{-7}$ 基本就稳了；大于 $10^{-3}$ 肯定有 bug。

这个习惯我后来一直保留——**任何新写的反向传播，先过梯度检验再训练**。这是工程上最值得的投资。

---

## 十二、特征工程：深度学习之前的传统视觉

`features.py` 是一份"前深度学习时代"的遗物，但理解它有助于看清神经网络到底替代了什么。

### HOG（方向梯度直方图）

1. 灰度化：$I_{gray} = 0.299R + 0.587G + 0.144B$
2. 用差分算梯度：$g_x, g_y$
3. 幅值与方向：$mag = \sqrt{g_x^2 + g_y^2},\ \theta = \arctan2(g_y, g_x)$
4. 把图像切成 8×8 cell，在每个 cell 内按 9 个方向 bin 累加幅值，得到直方图

HOG 在行人检测上曾红极一时（Dalal & Triggs, 2005）。它的核心思想是**用边缘方向分布描述局部外观**，比原始像素鲁棒得多。

### 颜色直方图（HSV）

转 HSV 后对 Hue 通道做直方图，捕捉图像的颜色分布。

### 思考

把 HOG + 颜色直方图喂给线性分类器，准确率能到 60%+，远超原始像素的 KNN。**这就是"特征工程"的价值**——人工设计好特征，分类器就轻松。而深度学习把这一步也交给模型去学：神经网络本质上就是一个**端到端的特征学习器**。

---

## 十三、Solver：把训练流程抽象出来

`solver.py` 是整个工程的"大脑"，它把训练循环和模型解耦：

- 每个 iteration：抽 mini-batch → `model.loss` 算 loss/grad → `update_rule` 更新参数
- 每个 epoch 末：学习率衰减 $\eta \leftarrow \eta \cdot lr\_decay$，检查 train/val accuracy
- 训练结束：保留验证集上最优的参数

```python
loss, grads = self.model.loss(X_batch, y_batch)
for p, w in self.model.params.items():
    dw = grads[p]
    next_w, next_config = self.update_rule(w, dw, config)
    self.model.params[p] = next_w
```

这种**模型/求解器分离**的设计，正是 PyTorch Lightning 等现代框架的雏形。

---

## 十四、数据预处理

`data_utils.py` 里的预处理看似简单，但每一步都有道理：

1. **均值减除**（零中心化）：$X \leftarrow X - \frac{1}{N}\sum_i X_i$
2. **NCHW 排布**：把 `(N, H, W, C)` 转成 `(N, C, H, W)`，方便卷积层处理

均值减除能让 loss surface 更圆，梯度下降更稳。后来才知道，BN 的第一步其实就是把这件事做得更彻底。

---

## 写在最后

做完 Assignment 1，我从"只会调 API"变成了"能从零实现反向传播"。最大的收获不是某个具体公式，而是几个**思维方式的转变**：

1. **向量化思维**——能用矩阵运算就别写循环
2. **模块化思维**——每层只管自己的前向/反向，组合起来就是网络
3. **验证思维**——梯度检验是反向传播的"单元测试"
4. **工程思维**——模型、求解器、优化器解耦，才好迭代实验

CS231n 最打动我的一句话是 Andrej Karpathy 说的：*"You should be able to implement backprop by hand."* 这份作业就是在逼你做到这件事。

下一站：Assignment 2 —— 卷积神经网络。期待。

---

> 参考资料：
> - [CS231n Course Notes](http://cs231n.github.io/)
> - [Batch Normalization 论文](https://arxiv.org/abs/1502.03167)
> - [HOG 论文（Dalal & Triggs, 2005）](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
> - 本仓库 `assignments/assignment1/cs231n/` 源码
