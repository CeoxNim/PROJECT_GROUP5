from keras.datasets import mnist
import scipy.misc
import os
import imageio
import numpy as np
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

class ImageData:
    def __init__(self, load_size, channels):
        self.load_size = load_size
        self.channels = channels

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        return img

def str2bool(x):
    return x.lower() in ('true')

'''加载数据集并预处理数据'''
def normalize(x) :
    return x/127.5 - 1
'''
加载并预处理mnist手写体数据集
数据预处理：normalize进行数据归一化处理，保证处理数据方便且收敛加快
            np.concentrate:数组的拼接
            np.random.shuffle:将数据集中的数据随机打乱顺序
            scipy.misc.imresize:将图像进行成比例放大或缩小，归一化到[0,255]区间，并将这些进行过放缩后的图像数组重新赋值给x，即相当于把原来的x中的每一个图像进行归一化放缩
            np.expand_dims：扩充数组最后一个参数
以上几个步骤对数据进行了预处理，包括：随机打乱图像的排列顺序，随机分出training data和testing data；对所有图像进行放缩处理
'''
def load_mnist(size=64):
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = normalize(train_data)
    test_data = normalize(test_data)
    x = np.concatenate((train_data, test_data), axis=0)
    np.random.seed(777)
    np.random.shuffle(x)
    x = np.asarray([np.array(Image.fromarray(x_img).resize((size, size))) for x_img in x])
    x = np.expand_dims(x, axis=-1)
    return x

'''
若是自定义的数据集：glob函数寻找指定路径中所有满足条件的文件
此处会找出./dataset目录下与dataset_name同名的文件夹中的所有图像数据
'''
def load_data(dataset_name, size=64) :
    x = glob(os.path.join("./dataset", dataset_name, '*.*'))
    return x

'''对图像数据进行预处理：从文件中读取图像作为数组（RGB色彩模式），缩放图像，归一化'''
def preprocessing(x, size):
    x = scipy.misc.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x




'''
本部分功能：根据RGB内容导出图片到指定路径下的文件夹，这里导出的图片相当于排列了多个结果图像的大画布
库imageio:读取照片RGB内容，转换照片格式，可导入各种格式类型的照片，再将其导出为各种格式的照片
imageio.imwrite:导出图片
merge:产生一个大的画布，使得图像排列好放置在相应位置，保存下这些图像
'''
def merge(images, size):
    h, w = images.shape[1], images.shape[2]   #get height&width of each images
    # numbers of channels
    #图片通道数=3or4：彩色图是三通道的；windows的bmp有时候是四通道的，加上一个通道表示透明度
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        #该循环使得images里的每张图片被放在指定的位置,(i,j)决定了每张图片再画布中的位置
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    #图片通道数=1：灰度图
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        #与上一种case的情况大致相同，但因为是灰度图只有一个通道数，因此在得到的表示整体画布的结果数组img中将第三个维度都赋成0即可
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')
def save_images(images, size, image_path):
    return imageio.imwrite(image_path, merge(inverse_transform(images), size))


'''检查log_dir文件夹是否存在于指定路径，若不存在则建立同名文件夹'''
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


'''
tf.trainable_variebles:仅可以查看可训练的变量
modol_analyzer:提供了分析tf图中的运算和变量的工具。此处打印出可训练的变量
'''
def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
'''将图像数据从[-1,1]变到[0，1]区间内'''
def inverse_transform(images):
    return (images+1.)/2.

