# LSGAN实现

## LSGAN简介

LSGAN（Least Squares Generative Adversarial Networks）是生成式对抗网络GAN的变体之一，继承了GAN的模型基本特征，即通过两个模块（Generative Model and Discriminative Model）的互相博弈学习产生比较好的输出。在*Least Squares Generative Adversarial Networks*这篇论文中首次提出了LSGAN模型。

### LSGAN基本原理

正如其名字所指示的，LSGAN的目标函数是一个平方误差，使用的是最小二乘函数，这与原始GAN所使用的交叉熵损失函数不同。在这种情况下，相应的目标函数将变成以下形式：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin_D+L%28D%29+%3D+%5Cmathbb%7BE%7D_%7Bx+%5Csim+p_x%7D+%28D%28x%29-b%29%5E2+%2B+%5Cmathbb%7BE%7D_%7Bz+%5Csim+p_z%7D+%28D%28G%28z%29%29-a%29%5E2)

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin_G+L%28G%29+%3D+%5Cmathbb%7BE%7D_%7Bz+%5Csim+p_z%7D+%28D%28G%28z%29%29-c%29%5E2)

论文通过证明指出（可以套用原始GAN的证明框架，具体证明过程参见论文）：当$b-c=1,b-a=2$时，目标函数等价于Pearson $\chi^2$ divergence，并且给出了两种初始化方案的选择：

Scheme I：$a=-1,b=1,c=0$

Scheme II：$a=0,b=c=1$

经过实验之后，作者说这两种方案的运行效果没有显著差异。因此，为了减少计算过程中的负数运算，我选择了Scheme II作为参数的初始化方案。

### LSGAN优势

根据论文内容，LSGAN相比原始GAN，大致有以下两个优势：

##### 		优化图片质量：

最小二乘损失函数会对处于判别成真的那些远离决策边界的样本进行惩罚，把远离决策边界的假样本拖进决策边界。

##### 		增强训练稳定性：

传统GAN的训练过程十分不稳定，在最小化目标函数时可能发生梯度弥散，很难更新生成器。 LSGAN 会惩罚那些远离决策边界的样本，这些样本的梯度是梯度下降的决定方向。

### LSGAN激活

在这里仅介绍我在训练LSGAN过程中所使用的两个激活函数：

###### 		Relu

$$
Relu(x)=max(0,x)=\left\{
\begin{array}{rcl}
x & {x \geq 0}\\
0 & {x < 0}
\end{array}
\right.
$$

函数图像如下：

![这里写图片描述](https://img-blog.csdn.net/20160917151806843)

###### 		Sigmoid

$$
Sigmoid(x) = \frac{1}{1 + e^{-x}}
$$

函数图像如下：

![这里写图片描述](https://img-blog.csdn.net/20160917150431945)

后续我会尝试使用另外两个激活函数leaky_relu，tanh作为替代，以比较效果是否有显著差异。

## LSGAN在MNIST数据集上的实现

根据小班老师的建议，我在LSGAN上使用了MNIST数据集来训练自己的模型。具体的代码在我的repo中。

搭建model的时候，参考了github上一些源码，发现大部分都是用Tensorflow或Pytorch实现模型构建，至于Chainer库的使用相对来说比较少。因此，我就试试引入Chainer库，来搭建自己的神经网络框架。后续有时间的话会尝试使用另外两个库模拟训练过程，比较效果差异。

在生成器的优化过程中，我使用了Adam优化器。

以下为训练结果

​	这是迭代10次的效果：（都是模糊的）

<img src="C:\Users\12288\AppData\Roaming\Typora\typora-user-images\image-20200612211856474.png" alt="image-20200612211856474"  />

​	这是迭代100次的效果：（有了数字的雏形轮廓了）

![image-20200612212126179](C:\Users\12288\AppData\Roaming\Typora\typora-user-images\image-20200612212126179.png)

​	这是迭代300次的效果：（虽然有些图片质量仍然不好，但还是能体现训练效果的）

![image-20200612212028175](C:\Users\12288\AppData\Roaming\Typora\typora-user-images\image-20200612212028175.png)

存在的问题：

1. 查资料的过程中，我发现大多数模型迭代次数在80~200之间就有很不错的效果，但是从我的实验结果来看，情况并不乐观。我觉得迭代次数过多依然是个很大的问题，这个值得后续改进（或许可以考虑使用其他的优化器，比如RMSProp；或者重新搭建神经网络；又或者可以考虑使用上述的另外两个激活函数；等等）。

2. 有篇文章[4]通过证明指出，LSGAN给出的损失函数并不符合WGAN前作的理论，LSGAN其实还是未能解决判别器足够优秀的时候，生成器还是会发生梯度弥散的问题，其原因是将判别器D的两个真假Loss加起来一并放入Adam中优化。这个问题我在训练过程中并没有发现，由于时间关系暂未发现原因。

## 主要引用文献或repo

[1]Mao, X., Li, Q., Xie, H., Lau, R. Y. K., & Wang, Z. (2016). Least Squares Generative Adversarial Networks, 1–15. [[1611.04076\] Least Squares Generative Adversarial Networks]

[2]https://github.com/pfnet-research/chainer-LSGAN

[3]https://github.com/musyoku/LSGAN

[4]LSGAN：最小二乘生成对抗网络 https://www.jiqizhixin.com/articles/2018-10-10-11

