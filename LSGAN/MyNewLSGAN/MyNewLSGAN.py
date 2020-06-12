import argparse
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from sklearn.datasets import fetch_mldata
from model import Discriminator, Generator

#get the argument
def parse_args():

    parser = argparse.ArgumentParser(description="Least Square Generative Adversarial Network")
    parser.add_argument("--num_epochs", '-n', type=int, default=500)
    parser.add_argument('--output', '-o', type=str, default="output_images")
    return parser.parse_args()

#load the mnist dataset from sklearn.datasets
def load_dataset():

    mnist = fetch_mldata("MNIST original", data_home="./MNIST_data")    #mnist dataset saved in "MNIST_data" dictionary
    x = np.float32(mnist.data) #minist.data:a 70000 × 784(28*28) array
    x /= np.max(x, axis=1, keepdims=True)   #normalization
    return x

#make a random noise sample 
def sample(n):

    noise_samples = np.float32(np.random.normal(0, 1, size=(n, 100)))
    return noise_samples

#print the image
def print_sample(filename, noise_samples, generator_):

    generated = generator_(np.asarray(noise_samples))
    images = generated.data

    for j, img in enumerate(images, start=1):
        plt.subplot(10, 10, j)    #set a 10*10 grid
        plt.imshow(img.reshape(28, 28), cmap="gray") #convert data to image
        plt.axis("off") 
    #Now we have finished a picture.

    plt.savefig(filename)    #save the image
    plt.clf()   #clear the image
    plt.close() #close the interface
    print("    Saved image to {}".format(filename))

#Plot the loss from each batch
def plotLoss(epoch, dLosses, gLosses):

    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/LSGAN_loss_epoch_{0:04}.png'.format(epoch))

def main(args):

    #initialize models and load mnist dataset
    G = Generator()
    D = Discriminator()
    x = load_dataset()

    #build optimizer of generator
    opt_generator = chainer.optimizers.Adam().setup(G)
    opt_generator.use_cleargrads()

    #build optimizer of discriminator
    opt_discriminator = chainer.optimizers.Adam().setup(D)
    opt_generator.use_cleargrads()

    #make the output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    #list of loss      
    Glosses = []
    Dlosses = []

    print("Now starting training loop...")

    #begin training process
    for train_iter in range(1, args.num_epochs + 1):

        for i in range(0, len(x), 100):

            #Clears all gradient arrays.
            #The following should be called before the backward computation at every iteration of the optimization.
            G.cleargrads()
            D.cleargrads()

            #Train the generator
            noise_samples = sample(100)
            Gloss = 0.5 * F.sum(F.square(D(G(np.asarray(noise_samples))) - 1))
            Gloss.backward()
            opt_generator.update()

            #As above
            G.cleargrads()
            D.cleargrads()

            #Train the discriminator
            noise_samples = sample(100)
            Dreal = D(np.asarray(x[i: i+100]))
            Dgen = D(G(np.asarray(noise_samples)))
            Dloss = 0.5 * F.sum(F.square((Dreal - 1.0))) + 0.5 * F.sum(F.square(Dgen))
            Dloss.backward()
            opt_discriminator.update()

        #save loss from each batch
        Glosses.append(Gloss.data)
        Dlosses.append(Dloss.data)

        if train_iter % 10 == 0:

            print("epoch {0:04d}".format(train_iter), end=", ")
            print("Gloss: {}".format(Gloss.data), end=", ")
            print("Dloss: {}".format(Dloss.data))

            noise_samples = sample(100)
            print_sample(os.path.join(args.output, "epoch_{0:04}.png".format(train_iter)), 
                         noise_samples, G)

    print("The training process is finished.")

    plotLoss(train_iter, Dlosses, Glosses)

if __name__=='__main__':
    args = parse_args()
    main(args)