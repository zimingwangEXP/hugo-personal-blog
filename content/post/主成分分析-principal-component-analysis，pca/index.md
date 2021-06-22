---
title: 主成分分析 ( Principal Component Analysis，PCA )
date: 2021-06-22T11:53:25.527Z
summary: ""
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
## PCA之前

机器学习包括有监督、无监督和强化学习。

无监督学习顾名思义即为无监督信号的条件下基于非配对数据的学习。如下图[1]所示，非配对数据无监督学习任务包含两种情况

1. 只有输入数据没有输出标签，即$D=\{X_1,X_2,...,X_n\}$。在这种数据下我们的目标往往是发现数据内蕴的任务相关的表达(表达学习),或者模式(模式识别), 或者特征(特征工程)。

2. 只有输出标签，而没有输入数据, 即$D=\{Y_1,Y_2,...,Y_n\}$。一般用于生成式任务，比如unconditional GAN从随机数字中生成$D$所描述的真实数据。

<img src="https://gitee.com/zi-ming-wang/img-cloud-pub/raw/master/image-20210620212715560.png" alt="image-20210620212715560" style="zoom: 50%;" />

对于第一种情况而言，其核心在于习得数据的内部结构、概率分布（或者隐变量）。

* 考虑习得数据的内部结构, 有Clustering
* 考虑习得数据的概率分布（隐变量）, 有Dimension Reduction、Matrix Factorization、Topic Model
