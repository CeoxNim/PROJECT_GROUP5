#### 丁乾坤SinGAN报告

（由于markdown格式限制，部分效果无法展现，丁乾坤SinGAN.ppt为最终完整版汇报）

##### 备注

由于训练模型太多太大，输出结果太多，无法上传到repo中，因此我将这些结果放到了北大网盘中，如果想要使用，可以直接从网盘中下载下来并放到相应的文件夹下，就可以进行操作，链接如下：[https://disk.pku.edu.cn:443/link/610184F51F6DCB4E694849E5C3F8C72F](https://disk.pku.edu.cn/link/610184F51F6DCB4E694849E5C3F8C72F) 有效期限：2020-07-11 23:59



##### 前言

SinGAN是输入一张图片并仅根据这一张图片进行训练的生成对抗网络。训练出来的GAN可以用于生成任意大小的以假乱真的照片，进行super resolution，生成animation等诸多操作。

我的主要工作包括文章内容简介与复现，用自己的数据评估文章提出的各项功能效果，根据结果理论分析SinGAN的实际效果并尝试给出未来继续发展的建议。主要的数据和分析在我更新过后的最新的ppt中。通过北大网盘可以下载我的生成数据和训练模型，下载后，生成的数据放在SinGAN-丁乾坤/SinGAN/Output目录底下，训练模型放在SinGAN-丁乾坤/SinGAN/TrainedModels中（由于数据太多，不能够全部放入总结ppt中）。



##### 报告内容提纲

1. 文章内容简介与文章内容复现（所有模型都需要自己训练，原文并没有给出预训练模型）

   - SinGAN结构简介

   - SinGAN文章提供的数据测试

   - 运行时间统计

2. 测试自己选择的数据，以测评SinGAN的实际效果

   - Generate Random Samples

   - Super Resolution

   - Generate Animation

   - MNIST组内对比

3. 针对SinGAN的理论分析，分析为什么在自己测试的数据上效果不佳，未来可能的改进方向。



##### 1 文章内容简介与复现

###### 1.1 SinGAN

SinGAN是输入一张图片并仅根据这一张图片进行训练的生成对抗网络。SinGAN可以通过这一张图片学习图像内部的分布规律。训练出来的GAN可以用于生成任意大小的以假乱真的照片，进行super resolution，生成animation等诸多操作。是一个多层次的结构，如下

![image-20200612153157243](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612153157243.png)



###### 1.2 文章数据训练与复现

首先，通过一张照片生成以假乱真的照片

![image-20200612153235278](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612153235278.png)

![image-20200612153454291](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612153454291.png)

其次，是生成.gif动画，具体效果请移步ppt

![image-20200612153538006](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612153538006.png)

训练时间

如果只用cpu训练的话，一张100KB大小的照片需要三天左右；如果借助gpu进行训练，一张100KB照片需要2小时左右，一张4KB照片需要10分钟左右 (MNIST)



##### 2 自己数据的测试

##### 2.1 Generate Random Samples

![image-20200612154026719](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154026719.png)

感觉…有点像iPad上面Photo Booth的旋转效果啊…

![image-20200612154124306](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154124306.png)

![image-20200612154136805](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154136805.png)

失败案例

![image-20200612154153200](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154153200.png)



##### 2.2 Super Resolution

下采样前：

![image-20200612154251628](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154251628.png)

下采样后：

![image-20200612154309736](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154309736.png)

下采样前：

![image-20200612154324295](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154324295.png)

下采样后：

![image-20200612154345652](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154345652.png)

![image-20200612154355097](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154355097.png)

![image-20200612154409734](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154409734.png)



###### 2.3 Animation

![image-20200612154435934](C:\Users\dqk\AppData\Roaming\Typora\typora-user-images\image-20200612154435934.png)



##### 3 对所有测试结果的理论分析

###### 3.1 Generate Random Samples

总的来说，对于每张输入的图片，SinGAN都生成至少50张图片，在这50张图片中一定还会有效果还可以的fake照片。尤其是针对自然景观等，SinGAN的效果十分显著。

但是，一旦图像中出现人，并且只有一两个人的时候，SinGAN的效果就会大打折扣。人的下半身会被忽略，人的面部会被扭曲等等。

毕竟，由于SinGAN是针对一张照片进行训练，所以对于照片的类型还是有一些要求的。需要画面中有许多重复的、相似的物体的照片，因为这样生成对抗网络才能够更好地学习internal distribution，否则学不到什么有意义的内容。

###### 3.2 Super Resolution

与generating random samples的结果不同，我对于SinGAN在super resolution方面的表现极其失望。无论是未经过下采样的照片还是经过了多次下采样的照片，SinGAN生成出来的高分辨率照片相比于原照片并没有明显提升，甚至还出现了诡异的彩色条纹和块状边界，模糊的数字并没有能够被恢复。

###### 3.3 Animation

和之前的分析类似，在有大量重复且类似形状物体的照片上，SinGAN的表现效果非常好，如闪电的动图。但是同样的，在人像上，尤其是单个人像上，效果非常不好，不能够保护原有的人的重要信息，面部以及身体会被扭曲。

###### 3.4 总结与未来改进方向

SinGAN由于是针对一张照片进行训练，所以对于照片的类型还是有较高要求的。画面中有许多重复的、相似的物体的照片是更好的选择。并且，由于kernel大小的限制，所有学习的特征最好不要特别大，像一个人这样的图片学习效果不会很好。

不能过度神话GAN的作用，神经网络也并不是万能的，每个领域都会一些不如专精于一个领域。关键的是解决问题的思路，神经网络更像是一个工具。文章作者提出了SinGAN可以解决很多CV领域的问题，但并不是每个领域都解决的很好，super resolution我认为效果实在不是很好。在未来，最好能够专注于发展某一种方法。

最后，训练GAN对算力的要求真高。



##### 参考文章

```
[ICCV 2019] SinGAN: Learning a Generative Model from a Single Natural Image
```

##### 





