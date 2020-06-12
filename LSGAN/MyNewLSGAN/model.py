import chainer
import chainer.links as L
import chainer.functions as F

#define the discriminator based on class Chain 
class Discriminator(chainer.Chain):

    def __init__(self):
        super(Discriminator, self).__init__(
            #make linear layer
            l1 = L.Linear(784, 400),
            l2 = L.Linear(400, 1))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        '''
            using relu as activation function:
            if x >= 0:
                relu(x) = x
            else
                relu(x) = 0
        '''
        return self.l2(h)

#define the generator based on class Chain 
class Generator(chainer.Chain):

    def __init__(self):
        super(Generator, self).__init__(
            #make linear layer
            l1 = L.Linear(100, 400),
            l2 = L.Linear(400, 784))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        return F.sigmoid(self.l2(h))
        '''
              using sigmoid as activation function:
              sigmoid(x) = 1 / (1+exp(-x))
        '''