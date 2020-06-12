'''
Description: A reproduction of acgan
Time: 2020.5.31
'''

import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
from scipy.stats import norm
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# My leaky_relu(alpha = 0.2)
def leaky_relu(x):
    return tf.where(tf.greater(x, 0), x, 0.2 * x)

# Build Generator
def generator(x):

    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0

    with tf.variable_scope('generator', reuse=reuse):
        x = slim.fully_connected(x, 1024)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
        x = slim.fully_connected(x, 7*7*128)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
        x = tf.reshape(x, [-1, 7, 7, 128])
        x = slim.conv2d_transpose(x, 64, kernel_size=[4, 4], stride=2, activation_fn=None)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
        z = slim.conv2d_transpose(x, 1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.sigmoid)
        z = tf.squeeze(z, -1)

    return z

# Build Discriminator
def discriminator(x, num_classes=10, num_cont=2):

    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0

    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = slim.conv2d(x, num_outputs=64, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)
        x = slim.conv2d(x, num_outputs=128, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)
        x = slim.flatten(x)
        shared_tensor = slim.fully_connected(x, num_outputs=1024, activation_fn=leaky_relu)
        recog_shared = slim.fully_connected(shared_tensor, num_outputs=128, activation_fn=leaky_relu)

        # Calculate the result of discrimination, category and content
        disc = slim.fully_connected(shared_tensor, num_outputs=1, activation_fn=None)
        disc = tf.squeeze(disc, -1)
        recog_cat = slim.fully_connected(recog_shared, num_outputs=num_classes, activation_fn=None)
        recog_cont = slim.fully_connected(recog_shared, num_outputs=num_cont, activation_fn=tf.nn.sigmoid)
        return disc, recog_cat, recog_cont

# Save the output images
def save_images(x):
    my_con = tf.placeholder(tf.float32, [batch_size, 2])
    myz = tf.concat(axis=1, values=[tf.one_hot(y, depth=classes_dim), my_con, z_rand])
    mygen = generator(myz)

    my_conl = np.ones([10, 2])
    a = np.linspace(-2, 2, 10)
    y_input = np.ones([10])
    figure = np.zeros((28 * 10, 28 * 10))
    my_rand = tf.random_normal((10, rand_dim))

    for i in range(10):
        for j in range(10):
            my_conl[j][0] = a[i]
            my_conl[j][1] = 0
            y_input[j] = j
        mygenoutv = sess.run(mygen, feed_dict={y:y_input, my_con:my_conl})
        for jj in range(10):
            digit = mygenoutv[jj].reshape(28, 28)
            figure[i*28:(i+1)*28, jj*28:(jj+1)*28] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(path + "/%d.png" % x)


# Set the const value
batch_size = 10 
classes_dim = 10
con_dim = 2
rand_dim = 38
n_input = 784

# Calculate
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, [None])

z_con = tf.random_normal((batch_size, con_dim))     # 2 cols
z_rand = tf.random_normal((batch_size, rand_dim))   # 38 cols
z = tf.concat(axis=1, values=[tf.one_hot(y, depth=classes_dim), z_con, z_rand])    # 50 cols

gen = generator(z)

y_real = tf.ones(batch_size)
y_fake = tf.zeros(batch_size)

disc_real, class_real, _ = discriminator(x)
disc_fake, class_fake, con_fake = discriminator(gen)
pred_class = tf.argmax(class_fake, dimension=1)

# ------------------
# Calculate the loss
# ------------------

# Discriminator loss
loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=y_real))
loss_d_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_fake))
loss_d = (loss_d_r + loss_d_f) / 2

# Generator loss
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_real))

# Class loss
loss_c_f = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_fake, labels=y))
loss_c_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_real, labels=y))
loss_c = (loss_c_r + loss_c_f) / 2

# Con loss
loss_con = tf.reduce_mean(tf.square(con_fake-z_con))


# Train the vars
t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]

disc_global_step = tf.Variable(0, trainable=False)
gen_global_step = tf.Variable(0, trainable=False)

deltaC = 7      # To emphasize the importance of the latent variables
train_disc = tf.train.AdamOptimizer(0.0001).minimize(
    loss_d + loss_c + loss_con * deltaC,
    var_list=d_vars,
    global_step=disc_global_step
)
train_gen = tf.train.AdamOptimizer(0.001).minimize(
    loss_g + loss_c + loss_con * deltaC,
    var_list=g_vars,
    global_step=gen_global_step
)

# Create a new folder to save the output images
path = 'acgan_images'
if os.path.exists(path):
    shutil.rmtree(path)
    os.mkdir(path)
else :
    os.mkdir(path)

# Run the sess and save the output image
training_epochs = 5
cnt = -1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x:batch_xs, y:batch_ys}
            l_disc, _, l_d_step = sess.run([loss_d, train_disc, disc_global_step], feeds)
            l_gen, _, l_g_step = sess.run([loss_g, train_gen, gen_global_step], feeds)
            
            cnt = cnt + 1
            if (cnt % 1000 == 0): 
                save_images(cnt)
        
    print('finish!')
    print('Result:', loss_d.eval({x:mnist.test.images[:batch_size], y:mnist.test.labels[:batch_size]}),
        loss_g.eval({x:mnist.test.images[:batch_size], y:mnist.test.labels[:batch_size]}))

    
