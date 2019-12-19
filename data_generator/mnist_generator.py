import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class DataGenerator:
    def __init__(self, config):

        self.config = config
        
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def next_batch(self, batch_size):
        x, y = self.mnist.train.next_batch(batch_size)
        x = np.reshape(x, (batch_size, 28, 28))
        #y = np.reshape(y, (batch_size, 10, 1))
        x = np.asarray(x)
        y = np.asarray(y)

        yield np.asarray(x), np.asarray(y)
    

