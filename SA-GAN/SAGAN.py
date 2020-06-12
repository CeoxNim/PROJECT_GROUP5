import time
from ops import *
from utils import *
from tensorflow.data.experimental import shuffle_and_repeat,prefetch_to_device, map_and_batch


'''
框架：构建模型-->训练模型-->training&test
'''

class SAGAN(object):
    def __init__(self, sess, args):
        self.model_name = "SAGAN"  # name for checkpoint
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.epoch = args.epoch
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.img_size = args.img_size
        #Generator
        self.layer_num = int(np.log2(self.img_size)) - 3
        self.z_dim = args.z_dim  # dimension of noise-vector
        self.gan_type = args.gan_type
        #Descriminator
        self.n_critic = args.n_critic
        self.sn = args.sn
        self.ld = args.ld
        self.sample_num = args.sample_num  # number of generated images to be saved
        self.test_num = args.test_num
        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.custom_dataset = False
        #dataset
        if self.dataset_name == 'mnist' :
            self.c_dim = 1
            self.data = load_mnist(size=self.img_size)
        else :
            self.c_dim = 3
            self.data = load_data(dataset_name=self.dataset_name, size=self.img_size)
            self.custom_dataset = True
        self.dataset_num = len(self.data)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

    '''
    Generator生成器模型：
    '''
    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            ch = 1024
            x = fully_connected(z, units=4 * 4 * ch, sn=self.sn, scope='fc')
            x = tf.reshape(x, [-1, 4, 4, ch])
            x = up_resblock(x, channels=ch, is_training=is_training, sn=self.sn, scope='front_resblock_0')
            for i in range(self.layer_num // 2) :
                x = up_resblock(x, channels=ch // 2, is_training=is_training, sn=self.sn, scope='middle_resblock_' + str(i))
                ch = ch // 2
            x = self.attention(x, channels=ch, scope='self_attention')
            for i in range(self.layer_num // 2, self.layer_num) :
                x = up_resblock(x, channels=ch // 2, is_training=is_training, sn=self.sn, scope='back_resblock_' + str(i))
                ch = ch // 2
            x = batch_norm(x, is_training)
            x = relu(x)
            x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, pad_type='reflect', scope='g_logit')
            x = tanh(x)
            return x


    '''
    Descriminator判别器模型：
    '''
    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = 64
            x = init_down_resblock(x, channels=ch, sn=self.sn, scope='init_resblock')
            x = down_resblock(x, channels=ch * 2, sn=self.sn, scope='front_down_resblock')
            x = self.attention(x, channels=ch * 2, scope='self_attention')
            ch = ch * 2
            for i in range(self.layer_num) :
                if i == self.layer_num - 1 :
                    x = down_resblock(x, channels=ch, sn=self.sn, to_down=False, scope='middle_down_resblock_' + str(i))
                else :
                    x = down_resblock(x, channels=ch * 2, sn=self.sn, scope='middle_down_resblock_' + str(i))
                ch = ch * 2
            x = lrelu(x, 0.2)
            x = global_sum_pooling(x)
            x = fully_connected(x, units=1, sn=self.sn, scope='d_logit')
            return x


    '''
    self-attention 机制
    '''
    def attention(self, x, channels, scope='attention'):
        with tf.variable_scope(scope):
            batch_size, height, width, num_channels = x.get_shape().as_list()
            #get f(x)、g(x)、h(x) in the figure
            f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')
            f = max_pooling(f)
            g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')
            h = conv(x, channels // 2, kernel=1, stride=1, sn=self.sn, scope='h_conv')
            h = max_pooling(h)
            #f(x)' * g(x)
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)
            #get attention map
            beta = tf.nn.softmax(s)
            #self-attention map
            o = tf.matmul(beta, hw_flatten(h))
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
            o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])
            o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
            x = gamma * o + x
        return x




    def build_model(self):
        #输入图像处理
        if self.custom_dataset :
            Image_Data_Class = ImageData(self.img_size, self.c_dim)
            inputs = tf.data.Dataset.from_tensor_slices(self.data)
            gpu_device = '/gpu:0'
            inputs = inputs.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
            inputs_iterator = inputs.make_one_shot_iterator()
            self.inputs = inputs_iterator.get_next()
        else :
            self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='real_images')
        #噪声
        self.z = tf.placeholder(tf.float32, [self.batch_size, 1, 1, self.z_dim], name='z')
        #损失函数
         # D:
        real_logits = self.discriminator(self.inputs)
        fake_images = self.generator(self.z)
        fake_logits = self.discriminator(fake_images, reuse=True)
        self.d_loss = discriminator_loss(self.gan_type, real=real_logits, fake=fake_logits)
         # G:
        self.g_loss = generator_loss(self.gan_type, fake=fake_logits)
        #训练training
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_vars)
        #测试testing
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)



    '''模型训练部分'''
    def train(self):
        tf.global_variables_initializer().run()
        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))
        #保存和加载模型函数
        self.saver = tf.train.Saver()
        #指定文件保存图
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        #check-point
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        for epoch in range(start_epoch, self.epoch):
            # 读取batch数据
            for idx in range(start_batch_id, self.iteration):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, 1, 1, self.z_dim])
                if self.custom_dataset :
                    train_feed_dict = {self.z: batch_z}
                else :
                    random_index = np.random.choice(self.dataset_num, size=self.batch_size, replace=False)
                    batch_images = self.data[random_index]
                    train_feed_dict = {self.inputs : batch_images,self.z : batch_z }
                # 更新D
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)
                # 更新
                g_loss = None
                if (counter - 1) % self.n_critic == 0:
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss
                # “进度条”
                counter += 1
                if g_loss == None :
                    g_loss = past_g_loss
                print("Epoch: [%2d] [%5d/%5d]  d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, d_loss, g_loss))

                #保存训练结果
                if np.mod(idx+1, self.print_freq) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                './' + self.sample_dir + '/' + self.model_name + '_train_{:02d}_{:05d}.png'.format(epoch, idx+1))
                if np.mod(idx+1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            start_batch_id = 0
            # 保存模型
            self.save(self.checkpoint_dir, counter)
        #保存模型（这里跟上一句不一样，是在最后一步的时候保存模型）
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type, self.img_size, self.z_dim, self.sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    #加载checkpiont
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))
        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    self.sample_dir + '/' + self.model_name + '_epoch%02d' % epoch + '_visualize.png')

    def test(self):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)
        #checkpoint
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        for i in range(self.test_num) :
            z_sample = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))
            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})
            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
                        [image_frame_dim, image_frame_dim],
                        result_dir + '/' + self.model_name + '_test_{}.png'.format(i))
