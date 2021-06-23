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
\ldots \\\\
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
\begin{aligned}
w_{2}=\underset{w}{\arg \max } w^{T} S w \\\\
\text { s.t. } \quad w^{T} w=1 \\\\
w_{1}^{T} w=0
\end{aligned}
$$
拉格朗日乘子法求解：
$$
\begin{aligned}
w_{2} &=\underset{w}{\arg \max }\left(w^{T} S w-\lambda\left(w^{T} w-1\right)-\beta w_{1}^{T} w\right) \\\\
\frac{\mathrm{d}\left(w^{T} S w-\lambda w^{T} w-\beta w_{1}^{T} w\right)}{\mathrm{d} w} &=2 w^{T} S^{T}-2 \lambda w^{T}-\beta w_{1}^{T}=0 \\\\
2 w^{T} S^{T}-2 \lambda w^{T}-\beta w_{1}^{T}=0 & \rightleftharpoons 2 S w-2 \lambda w-\beta w_{1}=0 \\\\
& \rightleftharpoons 2 w_{1}^{T} S w-2 \lambda w_{1}^T w-\beta w_{1}^{T} w_{1}=0 \\\\
& \rightleftharpoons 2\left(S^{T} w_{1}\right)^{T} w-0-\beta=0 \\\\
& \rightleftharpoons 2\left(S w_{1}\right)^{T} w-0-\beta=0 \\\\
& \rightleftharpoons 2\left(\lambda_{1} w_{1}\right)^{T} w-0-\beta=0 \\\\
& \rightleftharpoons \beta=0 \\\\
& \because \beta=0, \\\\
& \therefore 2 w^{T} S^{T}-2 \lambda w^{T}=0 \\\\
& \therefore S w=\lambda w
\end{aligned}
$$
易知$w_2$为与$w_1$正交的特征值第二大的特征向量

同理，易知$w_i$为特征值第$i$大的特征向量。因此按照降序对特征值排序，然后对每个特征值$\lambda_i$对应的特征子空间选$c_i$个单位正交基，所得到的基向量则组成了PCA的投影方向$\{w_1,w_2,...,w_m\},w_m \in R^n$。
#### 最小重构误差

PCA算法同样也可以从最小重构误差的角度获得

什么是重构？给一个投影点$y_i$，考虑将其在降维前的坐标系表达 
$$
\begin{array}{c}
\hat{x}_{i}=y_{i, 1} w_{1}+y_{i, 2} w_{2}+\ldots+y_{i, m} w_{m}=\left[w_{1}, w_{2}, \ldots, w_{n}\right] y_{i}=W y_{i}=W W^{T} x_{i}



\end{array}
$$





因此，最小化重构误差的最优化方程如下所示:
$$
\begin{aligned}
W &=\underset{W}{\arg \min } \frac{1}{p} \sum_{i=1}^{p}\left \Vert W W^{T} x_{i}-x_{i}\right \Vert^{2} \\\\
&=\underset{W}{\arg \min } \sum_{i=1}^{p}\left(x_{i}^{T}\left(W W^{T}-I\right)^{T}\left(W W^{T}-I\right) x_{i}\right) \\\\
&=\underset{W}{\arg \min } \sum_{i=1}^{p} x_{i}^{T}\left(-W W^{T}\right) x_{i} \\\\
&=\underset{W}{\arg \max } \sum_{i=1}^{p} \operatorname{tr}\left(x_{i}^{T} W W^{T} x_{i}\right) \\\\
&=\underset{W}{\arg \max } \sum_{i=1}^{p} \operatorname{tr}\left(W^{T} x_{i} x_{i}^{T} W\right) \\\\
&=\underset{W}{\arg \max } \sum_{i=1}^{p} \operatorname{tr}\left(W^{T} x_{i} x_{i}^{T} W\right) \\\\
&=\underset{W}{\arg \max } \operatorname{tr}\left(W^{T} \sum_{i=1}^{p}\left(x_{i} x_{i}^{T}\right) W\right) \\\\\
&=\underset{W}{\arg \max } \operatorname{tr}\left(W^{T} \hat{S} W\right) \\\\
&=\underset{W}{\arg \max } \sum_{i=1}^{m}\left(w_{i}^{T} \hat{S} w_{i}\right) \\\\
\text { s.t. } & W^{T} W=I \\\\
\text { where } \hat{S} &=\sum_{i=1}^{p}\left(x_{i} x_{i}^{T}\right)
\end{aligned}
$$
为了对比，重写原始逐投影向量的最大化方差目标函数为矩阵形式
$$
\begin{equation}
\begin{aligned}
\mathop{\arg\max}\limits_{w}w^TSw\Leftrightarrow &\mathop{\arg\max}\limits_{W}\sum_{i=1}^{m}w_i^TSw_i\Leftrightarrow \mathop{\arg\max}\limits_{W}tr(W^TSW)\\\\
&s.t. \quad W^TW =I \\\\
\end{aligned}
\end{equation}

$$
事实上，在正交单位的约束下, $cov(y_i,y_j)=w_i^TSw_j=w_i^T\lambda w_j = 0$，这意味着PCA降维以后达到了去关联的效果 ，因此上式中的$W^TSW$必为对角矩阵

<img src="https://gitee.com/zi-ming-wang/img-cloud-pub/raw/master/image-20210622120531411.png" alt="image-20210622120531411" style="zoom: 67%;" />

对比最大化方差和最小化重构误差的目标函数:
$$
\begin{array}{l}
\text { 最小重构误差: } \underset{W}{\arg \max } \sum_{i=1}^{m}\left(w_{i}^{T} \hat{S} w_{i}\right) \Leftrightarrow \underset{W}{\arg \max } \operatorname{tr}\left(W^{T} \hat{S} W\right)\\\\
\begin{array}{}
\text { 其中 } \hat{S}=\sum_{i=1}^{p}\left(x_{i} x_{i}^{T}\right) \\\\
\text { 最大化方差: } \underset{W}{\arg \max } \sum_{i=1}^{m} w_{i}^{T} S w_{i} \Leftrightarrow \underset{W}{\arg \max } \operatorname{tr}\left(W^{T} S W\right) \\\\
\text { 其中 } S=\left(\sum_{i=1}^{p}\left(x_{i}-\bar{x}\right)\left(x_{i}-\bar{x}\right)^{T}\right)
\end{array}
\end{array}
$$
我们可以看出两者之间唯一的区别在于中间的矩阵（散度）不同，当且仅当数据的均值为0时，求出的投影向量才完全相同。这也就是$\frac{x-\mu}{\sigma}$规范化中通过$x-\mu$去均值的意义所在，即去均值以后，得到的主成分（投影方向）不仅是最大化方差的，而且是最小化重构误差的。

> 矩阵表达下的最优化目标函数同样可以使用拉格朗日乘子法求解，区别只是从scalar对vector的求导变为对matrix的求导

#### PCA与SVD、Auto Encoder

PCA和SVD之间存在紧密的联系，事实上sklearn包（机器学习python库）中的PCA就是用SVD求的

<img src="https://gitee.com/zi-ming-wang/img-cloud-pub/raw/master/image-20210623094351911.png" alt="image-20210623094351911" style="zoom:50%;" />

重写最小重构误差的优化方程如下:
<div>
$$
\begin{array}{c}
W=\underset{W}{\arg \min } \frac{1}{p} \sum_{i=1}^{p}\left\|W W^{T} x_{i}-x_{i}\right\|^{2} \\\\
=\underset{W}{\arg \min } \frac{1}{p} \sum_{i=1}^{p}\left\|W v_{i}-x_{i}\right\|^{2} \\\\
\text { 其中 } v_{i}=W^{T} x_{i}
\end{array}
$$
</div>

将多个数据$x_i$拼在一起得到：
$$
\begin{array}{c}
W=\underset{W}{\arg \min }\Vert W V-X\Vert_{F}^{2} \\\\
X=\left\[x_{1}, x_{2}, \ldots, x_{p}\right\]^{T} \\\\
V=\left\[v_{1}, v_{2}, \ldots, v_{p}\right\]
\end{array}
$$
那么问题就转变成一个矩阵分解问题，我们希望对$X$进行分解成$WV$，使得两者的Frobenius范数最小，其中$W$是正交单的。"最小化Frobenius范数"，"正交", "矩阵分解"这些词很容易就联系到SVD。SVD告诉我们，对任何一个矩阵$X_{n\times m}$可以按照以下分解达成秩$m$下的最大逼近，其中$W_{n\times m}$为$XX^T$（在去均值的数据上，即为最大化方差的$S$，和最小化重构误差中的$\hat{S}$）的特征向量。因此从SVD的角度求投影向量$W$，只需要做SVD分解，然后求出前$m$大的奇异值所对应的左奇异向量即可。
$$
X_{n \times p}\approx W_{n \times m} \Sigma_{m \times m} V_{p \times m}^{T}
$$
从另一方面来看，当我们把最小重构误差的目标函数视损失函数，$x_i$视为标签，$WW^T$视为对输入$x_i$进行特征抽取的两层线性变化时，PCA就变成了一个特殊的无激活函数的双层线性Autoencoder。并且两层权重互为转置关系。

<img src="https://gitee.com/zi-ming-wang/img-cloud-pub/raw/master/image-20210622162008377.png" alt="image-20210622162008377" style="zoom:50%;" />

### 优点与局限

PCA是线性降维中在最大化方差和最小重构误差上最优的方法

然而，PCA也存在固有的缺点：

* PCA 不能解决高度非线性分布的数据降维问题，如球面
* PCA是无监督的算法，对于带标签数据无法充分利用标签信息，因此有了LDA
* PCA只能处理同一子空间下的降维问题，无法处理不同特征子空间样本降维问题，因此有了CCA

>  最大化方差或者最小重构误差是从线性投影的角度出发理解PCA算法的经典视角。但是从概率角度出发，即从发现数据中的隐变量出发，我们同样可以推导出PCA算法，也称之为PPCA。PPCA算法可以很好的与EM算法结合解决数据缺失的问题,并且可以自动确定数据的主子空间维度。本文面向初学者，故不对PPCA详加叙述，具体可参见Bishop的PRML。

##  代码

## Q&A

**这部分用于记录一些本人学习PCA时想的一些问题**

* 我们使用SVD的左奇异向量可以达到降维的作用，那么右奇异向量呢？

  如下式所示，SVD的左奇异向量用于压缩行，在$[x_1,x_2,...,x_n]$这种列布局的情况下即起到降维的作用

  SVD的左奇异向量用于压缩列，在$[x_1,x_2,...,x_n]$这种列布局的情况下即起到去除冗余样本的作用
$$
\begin{array}{c}
X_{n \times p} \approx W_{n \times m} \Sigma_{m \times m} V_{p \times m}^{T} \\\\
Y_{m \times p}=W_{n \times m}^{T} X_{n \times p} \\\\
Z_{n \times m}=X_{n \times p} V_{p \times m} \\\\
\text { where } m<n, m<p
\end{array}
$$





