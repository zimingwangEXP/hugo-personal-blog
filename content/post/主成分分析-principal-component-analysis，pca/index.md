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
<div>
$$
\begin{equation}
\begin{array}{c}
y_{i}=W^{T} x_{i}=\left[w_{1}, w_{2}, \ldots, w_{m}\right]^{T} x_{i}=\left[\begin{array}{c}
w_{1}^{T} \\
w_{2}^{T} \\
\ldots \\
w_{m}^{T}
\end{array}\right]=\left[\begin{array}{c}
w_{1}^{T} x_{i} \\
w_{2}^{T} x_{i} \\
\ldots \\
w_{m}^{T} x_{i}
\end{array}\right]=\left[\begin{array}{c}
<w_{1}, x_{i}> \\
<w_{2}, x_{i}> \\
\ldots \\
<w_{m}, x_{i}>
\end{array}\right] \\
W=\left[w_{1}, w_{2}, \ldots, w_{m}\right] \in R^{n \times m}
\end{array}
\end{equation}
$$</div>



* 一般而言，线性降维对投影向量有正交单位的约束条件（orthonormal）,即$W^TW=I$

<img src="https://gitee.com/zi-ming-wang/img-cloud-pub/raw/master/image-20210621003153792.png" alt="image-20210621003153792" style="zoom:40%;" />


