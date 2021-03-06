# Principal Component Aalysis (PCA) 简介

主成分分析（PCA)是一种线性的降维方法，通过将高维数据投影到低维子空间中提取信息。其目标在于保留数据更多变（多变意味着可分）的关键部分，丢弃其他变化较少、无关紧要的部分。

维度并不是什么神秘莫测的东西，可以说是表达数据的特征。比如说，一个大小为 28*28 的图片具有 784 个像素，这些像素就是共同表达此图片的特征，或者说 ——— 维度。

关于PCA，需要强调的一点是：它是一个无监督学习的降维方法，我们可以基于数据特征之间的相互关系对不同的数据进行归类，无需分类标签。

PCA是一种统计学方法，它使用 ***正交变换*** *(在线性代数中，正交变换是线性变换的一种，它从实内积空间V映射到V自身，且保证变换前后内积不变。因为向量的模长与夹角都是用内积定义的，所以正交变换前后一对向量各自的模长和它们的夹角都不变。)*  将一组可能相关的变量(每个变量都有不同的数值)的观测值转换成一组称为 ***主成分（Principal Component）*** 的 **线性不相关** 变量的值。 ( *Wikipedia* )
> **注意**: *特性* 、 *维度* 和 *变量* 是等价的，它们指的是同一种东西，可以替换使用。

<img src="http://www.sthda.com/english/sthda-upload/figures/principal-component-methods/006-principal-component-analysis-color-individuals-by-groups-and-variables-by-contributions-1.png" width = 50% height = 50% />

### 什么是“主成分”？
主成分是PCA算法的关键，它们表示隐藏在数据背后的内容。通俗地说，将数据从一个高维空间投影到一个较低维 —— 不妨假设是三维 —— 空间中时，这三个维度就是捕获/保留了数据大部分差异信息的三个“主成分”。

主成分含有 **大小** 和 **方向** 。方向表示数据在哪条 *"主轴"* 上最分散，或者说方差最大；大小表示PCA将数据投影到该 *"主轴"* 上后的方差。*第一主成分* 具有数据集的最大方差，“第二、第三 …… 主成分”的方差依次递减，以此类推，并且每个主成分都与前一个主成分正交。

通过这种方式，若给定一组x相关的y个样本，就能从这y个样本中得到一组u不相关的主成分。每个主成分代表着从数据中捕获的整体方差的一部分。

### PCA的使用场景有哪些？
+  **数据可视化**
<br>当研究数据相关问题时，如今的重要挑战是数据的绝对规模，以及定义数据的特性/维度/变量。在解决数据是关键的问题时，常常需要深入探索数据，比如找出变量之间的相互关系，或者理解变量的分布。
<br>考虑到复杂数据分布中有大量的变量/维度，对其进行可视化是一个巨大的，几乎不可能的挑战。

+  **对机器学习算法加速**
<br>既然PCA的主要思想是降维，如果数据维度过高，而机器学习算法的学习速度太慢，可以利用这PCA来加快机器学习算法的训练和测试速度。

## 数学表达

## PCA算法
```
**输入** ：
样本集D={x1, x2, ... , xm}; 低维空间维数d'
**过程** ：
+ 对所有样本进行 **中心化** ；
+ 计算样本的协方差矩阵；
+ 对协方差矩阵做特征值分解；
+ 取最大的d'个特征值所对应的特征向量w1, w2, ..., wd'.
**输出** ：
投影矩阵 W=(w1, w2, ..., wd')
```
<img src="http://pbs.twimg.com/media/DJaIzUKUEAAl6ga.jpg" width = 80% height = 80% />
