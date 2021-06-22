---
title: 主成分分析 ( Principal Component Analysis，PCA )
subtitle: 经典投影视角
date: 2021-06-22T11:53:25.527Z
summary: ""
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
{{< toc >}}
## PCA之前

机器学习包括有监督、无监督和强化学习。

无监督学习顾名思义即为无监督信号的条件下基于非配对数据的学习。如下图[1]所示，非配对数据无监督学习任务包含两种情况

1. 只有输入数据没有输出标签，即$D=\{X_1,X_2,...,X_n\}$。在这种数据下我们的目标往往是发现数据内蕴的任务相关的表达(表达学习),或者模式(模式识别), 或者特征(特征工程)。

2. 只有输出标签，而没有输入数据, 即$D=\{Y_1,Y_2,...,Y_n\}$。一般用于生成式任务，比如unconditional GAN从随机数字中生成$D$所描述的真实数据。

<img src="https://gitee.com/zi-ming-wang/img-cloud-pub/raw/master/image-20210620212715560.png" alt="image-20210620212715560" style="zoom: 50%;" />

对于第一种情况而言，其核心在于习得数据的内部结构、概率分布（或者隐变量）。

* 考虑习得数据的内部结构, 有Clustering
* 考虑习得数据的概率分布（隐变量）, 有Dimension Reduction、Matrix Factorization、Topic Model
## 原理 

### 一句话总结

**PCA是一种基于最大化方差或者最小重构误差原则的无监督线性降维算法**

### 问题定义

#### 什么是降维

* 前提: 数据点的原始空间表达存在冗余，有冗余才能降维，比如MNIST手写体数据集原始的数据表达空间有$28\times 28=784$维，但是我们知道这种784维向量表达实际上用于表达所有可能$28\times28$的灰度图，对于手写数字体而言，实际上数据的自由度（数据分布的流形维度）远小于$784$，极端情况下假设所有的手写数字体$\hat{\chi}_i$都能由一些基础的手写体$\chi_i$,通过水平平移$\Delta x$，垂直平移$\Delta x$，以及旋转$\Delta \theta$获得，那么事实上数据本身的自由度即为4，换句话来说我们通过$(i,\Delta x,\Delta y,\Delta \theta)$四维向量即可以表达原本$784$维空间下所有的手写数字体。当然实际的MNIST手写数字体本身的自由度考虑到粗细，扭曲等情况肯定不止$4$。但是即便用无监督线性PCA降维，仅使用$87$维的向量也可表达原始$784$维空间下$90\%$以上的信息。Anyway, 最重要的点在于只有当数据分布的流形维度是远低于数据原始空间维度的，换句话说数据点原始空间存在冗余，降维才有意义。

* 给定一组数据只有输入数据没有输出标签，即$D=\{x_1,x_2,...,x_p\}$, 共有$p$个数据点，每个数据点$x_i$是一个$n$维向量列向量，即 $x_i \in R^n$
* 通过一个变换$f$将每个$x_i$映射到一个$m$维向量$y_i$上, 即$y_i=f(W^Tx_i+b),y_i \in R^m, W \in R^{n\times m}$，其中$m \lt n$
* 变换$f$可以是线性变换，如PCA、CCA、LDA等等；也可以是非线性变换，如核方法，神经网络，流形学习等等
* 从特征工程的角度来看变换$f$
  * 对于特征选择（Feature Selection）而言，变换$f$是单位线性变换（恒等映射）, $b$为0向量，$W$是01矩阵 ，并且每列有且仅有一个1
  * 对于特征提取 (Feature Extraction) 而言，变换$f$通常为按照先验知识设计的变换，或者特别设计的神经网络（word embedding, graph embedding）
#### 线性降维

* 对于线性降维，不失一般性的令$f$为单位线性变换，$b$为零向量

* 如下图所示，问题即转换为需要找到一组投影向量$\{w_1,w_2,...,w_m\},w_m \in R^n$将原始数据点$x_i$投影到投影向量所张成的$m$维子空间中,得到投影点$y_i$。即

$$
\begin{array}{c}
y_{i}=W^{T} x_{i}=\left[w_{1}, w_{2}, \ldots, w_{m}\right]^{T} x_{i}=\left[\begin{array}{c}
w_{1}^{T}\\\\
w_{2}^{T} \\\\
\ldots \\\\
w_{m}^{T}
\end{array}\right]=\left[\begin{array}{c}
w_{1}^{T} x_{i} \\\\
w_{2}^{T} x_{i} \\\\
\ldots \\
w_{m}^{T} x_{i}
\end{array}\right]=\left[\begin{array}{c}
<w_{1}, x_{i}> \\\\
<w_{2}, x_{i}> \\\\
\ldots \\\\
<w_{m}, x_{i}>
\end{array}\right] \\\\
W=\left[w_{1}, w_{2}, \ldots, w_{m}\right] \in R^{n \times m}
\end{array}
$$




* 一般而言，线性降维对投影向量有正交单位的约束条件（orthonormal）,即$W^TW=I$

<img src="https://gitee.com/zi-ming-wang/img-cloud-pub/raw/master/image-20210621003153792.png" alt="image-20210621003153792" style="zoom:40%;" />
### PCA算法

> 理解算法背后的准则往往比理解算法本身更重要	

#### 最大方差	 

​	PCA算法从线性投影的角度来看所遵循的主要准则为最大方差原则。不严格的说，方差代表信息，最大方差即为最大化投影后的整体信息量。基于此准则，考虑到投影向量是正交单位的，我们可以逐投影向量的写出PCA的最优化问题，其中投影向量按投影方差排序，即$var(w_1^TX)\ge var(w_2^TX)\ge...\ge var(w_m^TX)$。

对于$w_1$，优化目标为
$$
\begin{array}{l}
w_{1} =\mathop{\arg\max}\limits_{w}\frac{1}{p-1}\sum_{i=1}^{p}(w^Tx_i-w^T\bar{x})^2  \\\\
\text{s.t.} \quad w^Tw =1 \\\\
\text{其中} \quad \bar{x} = \frac{1}{p}\sum_{i=1}^{p}x_i
\end{array}
$$
使用拉格朗日乘子法化简上述等式约束下的凸优化问题
$$
\begin{array}{l}
\begin{aligned}
w_{1} &=\underset{w}{\arg \max } \sum_{i=1}^{p}\left(w^{T} x_{i}-w^{T} \bar{x}\right)^{2}-\lambda\left(w^{T} w-1\right) \\\\
&=\underset{w}{\arg \max }\left(\sum_{i=1}^{p} w^{T}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{T} w\right)-\lambda w^{T} w \\\\
&=\underset{w}{\arg \max } w^{T}\left(\sum_{i=1}^{p}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{T}\right) w-\lambda w^{T} w \\\\
&=\underset{w}{\arg \max } w^{T} S w-\lambda w^{T} w
\end{aligned}\\\\
\text { 其中 } S=\left(\sum_{i=1}^{p}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{T}\right), \text { 且易知 } S \text { 为对称矩阵且半正定的 }
\end{array}
$$
令导数等0求解$w_1$（注：此处矩阵求导使用分子布局, 维度与前文保持一致）
$$
\begin{aligned}
\frac{\mathrm{d} (w^TSw-\lambda w^Tw )}{\mathrm{d} w} &=  \frac{\mathrm{d} (w^TSw)}{\mathrm{d} w}-\frac{(\lambda w^Tw )}{\mathrm{d} w} \\\\
&=\frac{\mathrm{d} w^T Sw+w^TS\mathrm{d}w}{\mathrm{d} w} -\lambda\frac{\mathrm{d} w^T w+w^T\mathrm{d}w}{\mathrm{d} w}\\\\
&=w^TS^T+w^TS-2\lambda w^T\\\\
&=2w^TS^T-2\lambda w^T\\\\
&=0 \\\\
\\\\
2w^TS^T-2\lambda w^T = 0&\rightleftharpoons  Sw=\lambda w\\\\
\end{aligned}
$$
$w$为非零向量,因此易知上述优化问题的解即投影向量$w_1$必然为协方差矩阵$S$的某一个特征向量。按照约定的规则$var(w_1^TX)\ge var(w_2^TX)\ge...\ge var(w_m^TX)$，$w_1$应为方差最大的特征向量,而由下式易知特征向量所对应的特征值$\lambda$即为投影后的方差。因此只需求出协方差矩阵$S$的$p$个特征值，然后从取最大特征值所对应的特征子空间选取一个非零的向量即得到$w_1$。
$$
var(w^TX) = w^TSw = \lambda w^Tw = \lambda
$$
下一个问题是，投影向量$w_1$求出来了，那么$w_2$,$w_3$,...$w_m$呢?

直观的来看，对于$w_2$同样有最大化方差的要求，因此和$w_1$一样必然为协方差矩阵$S$的某一个特征向量，但是按照正交性的要求，我们要求$w_2$必须和$w_1$正交。那么问题就转变成求一个$w_1$正交的特征向量，并且特征值尽可能大。我们知道对称矩阵$S$的$p$个特征值（算上重复的）是确定的，而每个特征值的几何重数（特征子空间的维度）等于其算数重数$c_i$，因此对于上述问题，必然是按照降序对特征值排序，然后对每个特征值$\lambda_i$对应的特征子空间选$c_i$个单位正交基，所得到的基向量必然为最优的投影方向$w_i$。

我们定义$w_2$的最优化方程：
$$
\begin{array}
w_2 =\mathop{\arg\max}\limits_{w}w^TSw \\\\
s.t. \quad
w^Tw =1\\\\
\quad \quad \quad w_1^Tw=0
\end{array}

$$
拉格朗日乘子法求解：
$$
\begin{aligned}
w_2 &=\mathop{\arg\max}\limits_{w}(w^TSw -\lambda(w^Tw-1)-\beta w_1^Tw) \\\\
\frac{\mathrm{d} (w^TSw-\lambda w^Tw-\beta w_1^Tw )}{\mathrm{d} w} &= 2w^TS^T-2\lambda w^T-\beta w_1^T = 0 \\\\
2w^TS^T-2\lambda w^T-\beta w_1^T = 0 &\rightleftharpoons 2Sw-2\lambda w-\beta w_1 = 0\\\\
&\rightleftharpoons 2w_1^TSw-2\lambda w_1^T w-\beta w_1^T w_1 =0\\\\
&\rightleftharpoons 2(S^Tw_1)^Tw-0-\beta=0\\\\
&\rightleftharpoons 2(Sw_1)^Tw-0-\beta=0 \\\\
&\rightleftharpoons 2(\lambda_1w_1)^Tw-0-\beta =0\\\\
&\rightleftharpoons \beta =0\\\\
\\\\
&\because \quad \beta = 0, \\\\& \therefore\quad  2w^TS^T-2\lambda w^T= 0

\\\\& \therefore\quad  Sw=\lambda w


\end{aligned}
$$
易知$w_2$为与$w_1$正交的特征值第二大的特征向量

同理，易知$w_i$为特征值第$i$大的特征向量。因此按照降序对特征值排序，然后对每个特征值$\lambda_i$对应的特征子空间选$c_i$个单位正交基，所得到的基向量则组成了PCA的投影方向$\{w_1,w_2,...,w_m\},w_m \in R^n$。


