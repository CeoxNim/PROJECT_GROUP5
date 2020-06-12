import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf_contrib.layers.xavier_initializer() #初始化权重矩阵
weight_regularizer = None
weight_regularizer_fully = None



'''
Layer:
    关于卷积、反卷积、全连接等基本操作的函数
    使用到tf.layers封装好的很多函数
'''
#卷积操作（关于卷积的内容见md文件）
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    #一些参数的解释：pad：扩充边缘，默认为0，不进行扩充
    #                stride:步长
    #                use_bias:是否要使用偏置向量
    #                sn:是否进行谱归一化
    with tf.variable_scope(scope):
        if pad > 0:
            #h--图像高度
            #注：只有tensor才可以使用get_shape()来获取张量大小，返回的是一个元组。as_list是将元组转换为list
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)
            #x的第2、3维度的填充方式
            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left
            #tf.pad:对张量在各个维度上进行填充。第一个维度是表示图像像素序列的序号，第四个维度表示图片的通道数，这两个维度不需要填充
            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            #reflect:定好边缘，按照边缘去对矩阵进行翻转，翻转的时候不复制边缘
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
        if sn==True:
            #tf.get_variable:创建卷积核这个新的变量，用weight_init(即权重矩阵)来初始化变量。可用于正则化
            #卷积核：[卷积核的高度，卷积核的宽度，卷积核个数，图像通道数]
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            #tf.nn.conv2d实现卷积操作，与tf.layers.conv2d不同的是，它需要自行传入初始化好的四维filters
            #filter：卷积核    ‘VALID’：不考虑边界，边缘不填充
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),strides=[1, stride, stride, 1], padding='VALID')
            if use_bias==True:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else:
            #tf.layers.conv2d实现卷积操作
            x = tf.layers.conv2d(inputs=x, filters=channels,kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,strides=stride, use_bias=use_bias)
        return x

#反卷积操作（转置卷积），卷积的反向操作，与上面的卷积操作函数有类似的讨论
def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        #padding：'SAME'or'VALID',决定了卷积的不同方式
        if padding == 'SAME':
            # output_shape:反卷积操作输出的shape
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),x_shape[2] * stride + max(kernel - stride, 0), channels]
        if sn==True:
            #filter：卷积核[卷积核的高度，卷积核的宽度，卷积核个数，图像通道数]
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,strides=[1, stride, stride, 1], padding=padding)
            if use_bias==True:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,strides=stride, padding=padding, use_bias=use_bias)
        return x

#展平一个tonsor
def flatten(x) :
    return tf.layers.flatten(x)

#将二维的feature展成一位的向量
def hw_flatten(x) :
    #将x转换为shape的形式（-1代表的是这一维度的大小由函数自行计算）
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

#全连接:将分布式特征representation映射到样本标记空间
def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]
        if sn==True:
            w = tf.get_variable("kernel", [channels, units], tf.float32,initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias==True:
                bias = tf.get_variable("bias", [units],initializer=tf.constant_initializer(0.0))
                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))
        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,kernel_regularizer=weight_regularizer_fully,use_bias=use_bias)
        return x




'''
Residual-block 残差块
'''
#
def up_resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = batch_norm(x_init, is_training)
            x = relu(x)
            x = up_sample(x, scale_factor=2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=sn)

        with tf.variable_scope('res2'):
            x = batch_norm(x, is_training)
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        with tf.variable_scope('shortcut'):
            x_init = up_sample(x_init, scale_factor=2)
            x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=False, sn=sn)

        return x + x_init

def down_resblock(x_init, channels, to_down=True, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        init_channel = x_init.shape.as_list()[-1]
        with tf.variable_scope('res1'):
            x = lrelu(x_init, 0.2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2'):
            x = lrelu(x, 0.2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

            if to_down :
                x = down_sample(x)

        if to_down or init_channel != channels :
            with tf.variable_scope('shortcut'):
                x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)
                if to_down :
                    x_init = down_sample(x_init)

        return x + x_init

def init_down_resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = lrelu(x, 0.2)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = down_sample(x)

        with tf.variable_scope('shortcut'):
            x_init = down_sample(x_init)
            x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)

        return x + x_init



'''
采样Sampling
'''
#维度上的均值
def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap
def global_sum_pooling(x) :
    gsp = tf.reduce_sum(x, axis=[1, 2])
    return gsp
#上采样：简单来说就是放大图片，使用临界点插值来完成
def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)
#下采样/降采样/图像缩小：使用均值池化来完成
def down_sample(x):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')
#最大池化层
def max_pooling(x) :
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')


'''
Normalization(归一化)相关函数
'''
#Batch Normalization
#可用作卷积和全连接操作的批正则化函数，根据当前批次数据按通道计算的均值和方差进行正则化。
def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,decay=0.9, epsilon=1e-05,center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

#谱归一化：使满足利普希茨连续性，限制了函数变化的剧烈程度，从而使模型更稳定。
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    #power iteration , usually 1 is enough
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm



'''
激活函数：activation function
'''
#Leaky ReLU激活函数
def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)
def relu(x):
    return tf.nn.relu(x)
def tanh(x):
    return tf.tanh(x)




'''
Loss function
'''
def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0
    if loss_func == 'gan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
    if loss_func == 'sagan' :
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))
    loss = real_loss + fake_loss
    return loss
def generator_loss(loss_func, fake):
    fake_loss = 0
    if loss_func == 'gan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))
    if loss_func == 'sagan' :
        fake_loss = -tf.reduce_mean(fake)
    loss = fake_loss
    return loss