# Self-attention GAN

the original paper:  [Self-Attention Generative Adversarial Networks.pdf](Self-Attention Generative Adversarial Networks.pdf) 



## Self-attention机制

传统的GAN：很容易学到纹理特征，如皮毛、天空、草地等，但不容易学习到特定的结构和几何特征

出于对于基本结构的最朴素的愿望——引入self-attention机制

### attention

​	attentio机制又称为注意力机制，顾名思义，是一种能让模型对重要信息重点关注并充分学习吸收的技术，它不算是一个完整的模型，应当是一种技术，能够作用于任何序列模型中。	

​	attention机制最早应用于图像领域，在上世纪九十年代就被提出，目前attention机制被广泛应用在基于RNN/CNN等神经网络模型的各种NLP任务中。在2017年，google机器翻译团队发表的《Attention is all you need》中大量使用了自注意力（self-attention）机制来学习文本表示，这篇论文也引起了很大的反应。

​	attention机制，简单来说，就是一系列注意力系数的分配。

### self-attention

​	self-attention其实就是attention机制的一种特殊情况，是需要计算自己对自己的权重。

![framework](C:\Users\Yan Lijun\Desktop\my project\images_in_md\framework.PNG)

​	f、g、h都是对feature maps做一个1*1卷积

​	f(x)' * g(x)得到一个表示feature maps 上各个像素点之间的相关性，将该相关性矩阵进行Softmax归一化后，得到attention map矩阵，代表了attention的方式。

​	h(x)*attention map：将attention应用到h(x)上，这样就得到了一个self-attention的map，可以得到每一个的像素点与feature map上其它点的相关性。

### 关于attention机制的参考博文

浅谈attention机制的理解：https://zhuanlan.zhihu.com/p/35571412

关于《注意力模型--Attention注意力机制》的学习https://www.jianshu.com/p/e14c6a722381

（以上两篇文章的举例、分析内容更偏向NLP方向，因为attention目前比较火热的应用方向是在机器翻译这一块）





## Samples & Results

### dataset

​	images of three celebrities

​	我自己的算力不够所以得到的效果不是特别好，因此我使用了跟一个博主相同的数据集，这个博主的结果的效果比我要好很多，接下来会展示一些生成的图像

### my_result

​	形成模糊的人脸，细节处仍然失真

​	(我的设备的算力确实不够，只能用CPU硬跑，这几天考试场次太多电脑空不下来，没有办法跑一整天，这幅图的结果是花了大约十二小时)

![result 2](C:\Users\Yan Lijun\Desktop\my project\images_in_md\result 2.png)

### result from a blog

​	这是网络上一个博主同样使用celebA的结果，可以看到，在算力足够的条件下，SA-GAN的效果还是很不错的。

![celebA](C:\Users\Yan Lijun\Desktop\my project\images_in_md\celebA.png)







## Code

​	一共有四个python file，每个python file中实现的具体功能以及函数的作用等，都在代码中直接添加了注释，可以直接到每个文件中去阅读代码和注释，因此此处只简要介绍每个python file的总体功能。

​	utils:对数据集的处理、对生成图像的导出等一些实用工具

​	ops：卷积、反卷积、残差、激活函数、归一化函数

​	SAGAN：GAN的主体部分，包括generator、discriminator、attention等

​	main：入口

其中：

​	utils、ops参考了https://github.com/taki0112/Tensorflow-Cookbook，是直接套用的一些现成的“工具”，例如卷积、输入预处理等

​	main的参数设置这一部分，引用了https://github.com/taki0112/StarGAN-Tensorflow/blob/master/main.py的内容（说实话输入参数设置这方面比较懵）



## 如何使用代码？	

### train

​	python main.py --phase train --dataset celebA 

​		(celebA可更改，改为其他的自定义数据集即可)

### test

​	python main.py --phase test --dataset celebA 